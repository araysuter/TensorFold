//! Private gateway control routes, enabled only by TF_SESSION_ROOT and protected by the backend's own bearer key.
const std = @import("std");
const sys = @import("sys");
const api = @import("engine_api");
const json = @import("json");
const http_body = @import("http_body.zig");
const routes = @import("routes.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;
pub const prefix = "/v1/internal/session/";
var controlling: std.atomic.Value(bool) = .init(false);

pub fn post(srv: *Server, conn: *Conn, a: Allocator, route: []const u8) void {
    const root = if (sys.c.getenv("TF_SESSION_ROOT")) |p| std.mem.span(p) else {
        _ = http_body.read(conn, a, http_body.limit) catch {};
        return routes.unknown(conn, a);
    };
    if (!conn.auth_enabled or conn.key_label == null) {
        _ = http_body.read(conn, a, http_body.limit) catch {};
        return fail(conn, a, 403, "session controls require backend bearer authentication");
    }
    if (controlling.swap(true, .seq_cst)) {
        _ = http_body.read(conn, a, http_body.limit) catch {};
        return fail(conn, a, 503, "another session control is in progress");
    }
    defer controlling.store(false, .seq_cst);
    _ = srv.generating.fetchAdd(1, .seq_cst);
    defer _ = srv.generating.fetchSub(1, .seq_cst);
    if (srv.draining.load(.seq_cst)) return routes.draining(conn, false);
    const op: api.ImprintOp = if (std.mem.eql(u8, route, prefix ++ "save")) .session_save else if (std.mem.eql(u8, route, prefix ++ "load")) .session_load else if (std.mem.eql(u8, route, prefix ++ "clear")) .session_clear else {
        _ = http_body.read(conn, a, http_body.limit) catch {};
        return routes.unknown(conn, a);
    };
    const read = http_body.read(conn, a, 4096) catch return fail(conn, a, 400, "invalid session request");
    const bytes = switch (read) { .ok => |b| b, .refused => return fail(conn, a, 400, "invalid session request") };
    const parsed = json.parse(a, bytes) catch return fail(conn, a, 400, "invalid session JSON");
    const value = switch (parsed) { .ok => |v| v, .err => return fail(conn, a, 400, "invalid session JSON") };
    if (value != .object) return fail(conn, a, 400, "session request must be an object");
    var path: []const u8 = "";
    if (op != .session_clear) {
        const v = value.get("path") orelse return fail(conn, a, 400, "path is required");
        if (v != .string or !api.session_file.permitted(root, v.string)) return fail(conn, a, 400, "path must be a flat .tfs file under TF_SESSION_ROOT");
        path = v.string;
    }
    const ask = srv.gpa.create(api.ImprintAsk) catch return fail(conn, a, 500, "out of memory");
    ask.* = .{ .op = op };
    const p = srv.gpa.dupeSentinel(u8, path, 0) catch {
        srv.gpa.destroy(ask);
        return fail(conn, a, 500, "out of memory");
    };
    ask.session_path = p;
    // The preceding handler emits its final body/[DONE] before enqueueing the canonical warm pass.

    // Its generating count is held until that enqueue finishes; the private gateway serializes controls.
    var waited: u32 = 0;
    while (srv.generating.load(.seq_cst) > 1 and waited < 10 * 60 * 1000) : (waited += 2) {
        std.Io.sleep(srv.io, .fromMilliseconds(2), .awake) catch {};
    }
    if (srv.generating.load(.seq_cst) > 1) {
        srv.gpa.free(p);
        srv.gpa.destroy(ask);
        return fail(conn, a, 503, "preceding generation handler did not finish");
    }
    if (!srv.engine.imprintAsk(ask)) {
        srv.gpa.free(p);
        srv.gpa.destroy(ask);
        return fail(conn, a, 503, "session engine unavailable");
    }
    // Control ops wait for all foreground work and the canonical next-turn warm pass to end.
    if (!ask.wait(srv.io, 10 * 60 * 1000)) return fail(conn, a, 503, "session control timeout");
    defer srv.gpa.free(p);
    defer srv.gpa.destroy(ask);
    if (ask.session_error) |err| return fail(conn, a, 409, @errorName(err));
    const o = json.newObject(a) catch return fail(conn, a, 500, "out of memory");
    o.put(a, "object", .{ .string = switch (op) { .session_save => "session.save", .session_load => "session.load", .session_clear => "session.clear", else => unreachable } }) catch return;
    o.put(a, "tokens", json.intValue(a, ask.session_receipt.tokens) catch return) catch return;
    o.put(a, "bytes", json.intValue(a, ask.session_receipt.bytes) catch return) catch return;
    const id = std.fmt.allocPrint(a, "{x:0>16}", .{ask.session_receipt.fingerprint}) catch return;
    o.put(a, "fingerprint", .{ .string = id }) catch return;
    routes.sendValue(conn, a, 200, .{ .object = o });
}

fn fail(conn: *Conn, a: Allocator, status: u16, message: []const u8) void {
    const quoted = json.quote(a, message, .{}) catch return;
    const body = std.fmt.allocPrint(a, "{{\"error\":{{\"message\":{s},\"type\":\"snapshot_error\",\"code\":\"snapshot_error\"}}}}", .{quoted}) catch return;
    conn.sendJson(status, body);
}
