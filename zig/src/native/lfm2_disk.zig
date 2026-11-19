//! LFM2 target and DSpark kept bytes through the shared checksummed conversation store.
const std = @import("std");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.lfm2_moe;
const rows = api.prompt_cache.rows;
const Host = @import("lfm2_moe.zig").Host;
const Snap = q.snapshot.Snap;
const Allocator = std.mem.Allocator;

pub const image: rows.Image = .{ .streamed_reads = true, .parts = parts, .gather = gather, .blank = blank, .scatter = scatter };
fn host(ptr: *anyopaque) *Host {
    return @ptrCast(@alignCast(ptr));
}
fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) !void {
    const h = host(ptr);
    try out.append(a, .{ .bytes = 8 });
    try out.append(a, .{ .bytes = q.backend.Snaps.bytes(h.metal, at) });
    if (h.draft) |d| try out.append(a, .{ .bytes = d.keptBytes(at) });
}
fn blank(ptr: *anyopaque, at: u32) !api.prompt_cache.Saved {
    const h = host(ptr);
    const s = try h.gpa.create(Snap);
    errdefer h.gpa.destroy(s);
    s.* = .{ .at = at, .draft_usable = false, .bytes = try h.gpa.alloc(u8, @intCast(q.backend.Snaps.bytes(h.metal, at))) };
    errdefer h.gpa.free(s.bytes);
    if (h.draft) |d| s.draft = try h.gpa.alloc(u8, @intCast(d.keptBytes(at)));
    return s;
}
fn imageIO(h: *Host, s: *Snap, ranges: []const rows.image.Range, buffer: []u8, read: bool) !void {
    var header: [8]u8 = undefined;
    std.mem.writeInt(u64, &header, @intFromBool(s.draft != null and s.draft_usable), .little);
    const draft_bytes: usize = if (h.draft) |d| @intCast(d.keptBytes(s.at)) else 0;
    const planes = [_]q.snapshot_image.Plane{
        .{ .data = &header, .len = header.len },  .{ .data = s.bytes, .len = s.bytes.len },
        .{ .data = s.draft, .len = draft_bytes },
    };
    try q.snapshot_image.transferPlanes(&planes, ranges, buffer, read);
    if (!read) {
        const flag = std.mem.readInt(u64, &header, .little);
        if (flag > 1) return error.BadLfmSnapshotHeader;
        s.draft_usable = flag == 1;
    }
}
fn gather(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) !void {
    try imageIO(host(ptr), @ptrCast(@alignCast(saved)), ranges, out, true);
}
fn scatter(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) !void {
    try imageIO(host(ptr), @ptrCast(@alignCast(saved)), ranges, @constCast(in), false);
}

pub fn attach(h: *Host, a: Allocator, io: std.Io, o: api.Open) !void {
    const root = o.learn orelse return error.LfmDiskRootMissing;
    const store = &(h.cache orelse return error.LfmDiskNeedsPromptCache);
    var hash = std.hash.Wyhash.init(0x6c666d32);
    try @import("sys").file_identity.current(&hash, a, io);
    const identity = @import("disk_identity.zig");
    try identity.checkpoint(&hash, a, io, o.dir);
    if (o.drafter) |dir| try identity.checkpoint(&hash, a, io, dir);
    identity.machine(&hash, std.mem.span(h.model.device.name()));
    identity.profile(&hash, .{ .family = "lfm2_moe", .backend = "metal" });
    const words = [_]u64{ 2, h.host.info_.context_window, h.metal.options.chunk, @intFromBool(h.draft != null), o.drafter_bits, h.metal.width };
    hash.update(std.mem.sliceAsBytes(&words));
    h.disk = try rows.attach(h.gpa, store, .{
        .root = root,
        .identity = hash.final(),
        .cap = @intFromFloat(o.learn_gib * (1 << 30)),
        .floor = @intFromFloat(o.learn_min_free_gib * (1 << 30)),
        .daily = @intFromFloat(o.learn_daily_gib * (1 << 30)),
        .lookahead = store.rules.lookahead,
        .planned = store.rules.planned,
        .grid = store.rules.grid,
        .spill = .evicted,
        .sync_writes = true,
        .slots = @import("disk_spill.zig").slots(o),
        .longest = h.host.info_.context_window,
    });
}
