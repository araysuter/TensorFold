//! Explicit complete native conversation snapshots for a serial model gateway.

//! Writes bounded pieces, atomically commits a 0600 file, validates every byte before publishing a restore.
const std = @import("std");
const sys = @import("sys");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const image = @import("rows_image.zig");
const file = @import("rows_file.zig");
const modes = @import("cache_modes.zig");
const Allocator = std.mem.Allocator;
const piece: usize = 16 << 20;
const mark: u32 = 0x53534654; // TFSS: this is not an automatic learned-row state.
const VERSION: u32 = 1;
const Header = extern struct { magic: u32 = mark, version: u32 = VERSION, tokens: u32, spans: u32, born: u32, lookahead: u32, chunk: u32, planned: u32 };
pub const Receipt = struct { tokens: u32 = 0, bytes: u64 = 0, fingerprint: u64 = 0 };

fn fingerprint(s: *const pc.Store) !u64 { return s.session_identity orelse error.SessionsDisabled; }
fn name(id: u64) [32]u8 {
    var out: [32]u8 = undefined;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("TensorFold gateway session codec v1");
    h.update(std.mem.asBytes(&id));
    h.final(&out);
    return out;
}

/// Only one flat, ordinary filename under the private root; no symlink components are accepted by the gateway.
pub fn permitted(root: []const u8, path: []const u8) bool {
    if (root.len < 2 or root[0] != '/' or std.mem.endsWith(u8, root, "/")) return false;
    if (!std.mem.startsWith(u8, path, root) or path.len <= root.len + 5 or path[root.len] != '/') return false;
    const leaf = path[root.len + 1 ..];
    if (!std.mem.endsWith(u8, leaf, ".tfs") or leaf.len > 160 or leaf[0] == '.') return false;
    for (leaf) |c| if (!std.ascii.isAlphanumeric(c) and c != '-' and c != '_' and c != '.') return false;
    return true;
}

fn latest(s: *pc.Store) ?*pc.Entry {
    var result: ?*pc.Entry = null;
    for (s.entries.items) |e| if (!e.expired) {
        if (result == null or e.used > result.?.used or (e.used == result.?.used and e.at > result.?.at)) result = e;
    };
    return result;
}

fn parts(s: *pc.Store, at: u32, a: Allocator) ![]image.Part {
    const img = s.family.vtable.rows orelse return error.SessionsUnsupported;
    var list: std.ArrayList(image.Part) = .empty;
    errdefer list.deinit(a);
    try img.parts(s.family.ptr, at, &list, a);
    return list.toOwnedSlice(a);
}

pub fn save(s: *pc.Store, path: []const u8) !Receipt {
    const id = try fingerprint(s);
    const e = latest(s) orelse return .{ .fingerprint = id };
    if (e.borrowers != 0) return error.SessionBusy;
    const img = s.family.vtable.rows orelse return error.SessionsUnsupported;
    if (img.whole) |whole| if (!whole(s.family.ptr, e.saved)) return error.SessionIncomplete;
    const ps = try parts(s, e.at, s.gpa);
    defer s.gpa.free(ps);
    const scratch = try s.gpa.alloc(u8, piece);
    defer s.gpa.free(scratch);
    const dir = std.fs.path.dirname(path) orelse return error.SessionPath;
    const leaf = std.fs.path.basename(path);
    var w = try file.Writer.open(dir, leaf, "");
    var done = false;
    defer if (!done) w.abort();
    var off: u64 = 0;
    for (ps) |p| {
        var at: u64 = 0;
        while (at < p.bytes) {
            const n: usize = @intCast(@min(piece, p.bytes - at));
            try img.gather(s.family.ptr, e.saved, &.{.{ .off = off + at, .len = n }}, scratch[0..n]);
            try w.append(scratch[0..n]);
            at += n;
        }
        off += p.bytes;
    }
    const h: Header = .{ .tokens = @intCast(e.tokens.len), .spans = @intCast(e.decode_spans.len), .born = e.born, .lookahead = s.rules.lookahead, .chunk = s.rules.chunk, .planned = @intFromBool(s.rules.planned) };
    const n = try w.finish(.{ .kind = @backingInt(file.Kind.state), .flags = mark, .payload = 0, .payload_hash = 0, .name = name(id), .at = e.at, .block = VERSION }, &.{ std.mem.asBytes(&h), std.mem.sliceAsBytes(e.tokens), std.mem.sliceAsBytes(e.decode_spans) }, true);
    done = true;
    return .{ .tokens = e.at, .bytes = n, .fingerprint = id };
}

/// Read without a whole-state allocation: macOS backend only, NOFOLLOW and NOCACHE set by file.open.
fn readAt(fd: c_int, out: []u8, at: u64) !void {
    var used: usize = 0;
    while (used < out.len) {
        const n = sys.c.pread(fd, out[used..].ptr, out.len - used, @intCast(at + used));
        if (n < 0 and sys.c.errno(n) == .INTR) continue;
        if (n <= 0) return error.SessionRead;
        used += @intCast(n);
    }
}

/// Import into one kept RAM state. The live runner restores it through the ordinary exact token/mode lookup.

/// Any mismatch or corrupt byte fails; a missing or failed restore never falls back to a fresh prefill.
pub fn load(s: *pc.Store, path: [:0]const u8) !Receipt {
    const id = try fingerprint(s);
    var opened = try file.open(path, .state, name(id));
    defer opened.close();
    const t = opened.trailer;
    if (t.flags != mark or t.block != VERSION or t.at == 0 or t.at > 131072 or t.meta < @sizeOf(Header) or t.meta > (4 << 20) or t.reserved != 0) return error.SessionFormat;
    const m = try file.meta(&opened, s.gpa);
    defer s.gpa.free(m);
    var h: Header = undefined;
    @memcpy(std.mem.asBytes(&h), m[0..@sizeOf(Header)]);
    if (h.magic != mark or h.version != VERSION or h.lookahead != s.rules.lookahead or h.chunk != s.rules.chunk or h.planned != @intFromBool(s.rules.planned)) return error.SessionFormat;
    if (h.tokens != t.at + s.rules.lookahead or h.spans > h.tokens or m.len != @sizeOf(Header) + @as(u64, h.tokens) * 4 + @as(u64, h.spans) * 8) return error.SessionFormat;
    const bytes = s.family.vtable.bytes(s.family.ptr, t.at);
    if (bytes > s.budget or s.most == 0) return error.SessionOverBudget;
    if (s.entries.items.len > 0) return error.SessionMustClear; // loading never doubles a 128K RAM snapshot.
    const img = s.family.vtable.rows orelse return error.SessionsUnsupported;
    const ps = try parts(s, t.at, s.gpa);
    defer s.gpa.free(ps);
    var expected: u64 = 0;
    for (ps) |p| expected += p.bytes;
    if (expected != t.payload) return error.SessionLayout;
    const tokens = try s.gpa.alloc(u32, h.tokens);
    errdefer s.gpa.free(tokens);
    @memcpy(std.mem.sliceAsBytes(tokens), m[@sizeOf(Header)..][0 .. @as(usize, h.tokens) * 4]);
    const spans = try s.gpa.alloc(modes.Span, h.spans);
    errdefer s.gpa.free(spans);
    @memcpy(std.mem.sliceAsBytes(spans), m[@sizeOf(Header) + @as(usize, h.tokens) * 4 ..]);
    try modes.validate(spans);
    for (spans) |span| if (span[1] > t.at) return error.SessionFormat;
    const last = try s.gpa.dupe(u32, tokens);
    errdefer s.gpa.free(last);
    const saved = try img.blank(s.family.ptr, t.at);
    errdefer s.family.vtable.drop(s.family.ptr, saved);
    const scratch = try s.gpa.alloc(u8, piece);
    defer s.gpa.free(scratch);
    var hash: std.hash.Wyhash = .init(0);
    var off: u64 = 0;
    for (ps) |p| {
        var at: u64 = 0;
        while (at < p.bytes) {
            const n: usize = @intCast(@min(piece, p.bytes - at));
            try readAt(opened.fd, scratch[0..n], off + at);
            hash.update(scratch[0..n]);
            try img.scatter(s.family.ptr, saved, &.{.{ .off = off + at, .len = n }}, scratch[0..n]);
            at += n;
        }
        off += p.bytes;
    }
    if (hash.final() != t.payload_hash) return error.SessionHash;
    const e = try s.gpa.create(pc.Entry);
    errdefer s.gpa.destroy(e);
    s.clock += 1;
    e.* = .{ .tokens = tokens, .decode_spans = spans, .at = t.at, .saved = saved, .bytes = bytes, .born = h.born, .used = s.clock, .last = last };
    try s.entries.append(s.gpa, e);
    s.held += bytes;
    s.session_required = saved;
    return .{ .tokens = t.at, .bytes = t.payload + t.meta + @sizeOf(file.Trailer), .fingerprint = id };
}

pub fn clear(s: *pc.Store) !Receipt {
    const id = try fingerprint(s);
    for (s.entries.items) |e| if (e.borrowers != 0) return error.SessionBusy;
    const n = s.held;
    const at = if (latest(s)) |e| e.at else 0;
    s.session_required = null;
    while (s.entries.items.len > 0) s.remove(s.entries.items.len - 1);
    s.shared_keys.clearRetainingCapacity();
    s.active_rewind_at = 0;
    if (s.family.vtable.trim) |trim| trim(s.family.ptr, 0);
    return .{ .tokens = at, .bytes = n, .fingerprint = id };
}

test "session paths are flat private files and reject traversal, root prefixes and hidden names" {
    try std.testing.expect(permitted("/tmp/sessions", "/tmp/sessions/ab-12_model.tfs"));
    for ([_][]const u8{ "/tmp/sessions2/a.tfs", "/tmp/sessions/../a.tfs", "/tmp/sessions/.a.tfs", "/tmp/sessions/a/b.tfs", "/tmp/sessions/a", "/tmp/sessions/a.tfs/", "/tmp/sessions/a%2fb.tfs" }) |p| try std.testing.expect(!permitted("/tmp/sessions", p));
}

test { _ = @import("session_file_test.zig"); }
