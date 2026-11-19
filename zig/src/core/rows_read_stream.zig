//! Opt-in CPU snapshot images read in bounded pieces into an unpublished blank state.
const std = @import("std");
const file = @import("rows_file.zig");
const image = @import("rows_image.zig");
const pc = @import("prompt_cache.zig");
const Rows = @import("rows_store.zig").Rows;

pub const piece_bytes: u64 = 16 << 20;

pub fn window(a: std.mem.Allocator, ranges: []const image.Range, offset: u64, len: u64) ![]image.Range {
    var out: std.ArrayList(image.Range) = .empty;
    errdefer out.deinit(a);
    var skip = offset;
    var left = len;
    for (ranges) |range| {
        if (skip >= range.len) {
            skip -= range.len;
            continue;
        }
        const n = @min(left, range.len - skip);
        try out.append(a, .{ .off = std.math.add(u64, range.off, skip) catch return error.RowsRead, .len = n });
        left -= n;
        skip = 0;
        if (left == 0) break;
    }
    if (left != 0) return error.RowsRead;
    return out.toOwnedSlice(a);
}

pub fn payload(r: *Rows, opened: *const file.Opened, ranges: []const image.Range, saved: pc.Saved) !void {
    const n = image.total(ranges);
    const stage = try r.staged(@min(n, piece_bytes));
    var hash = std.hash.Wyhash.init(0);
    var offset: u64 = 0;
    while (offset < n) {
        const count: usize = @intCast(@min(n - offset, piece_bytes));
        const bytes = stage[0..count];
        if (!file.readUnaligned(opened.fd, bytes, offset)) return error.RowsRead;
        hash.update(bytes);
        const slices = try window(r.gpa, ranges, offset, count);
        defer r.gpa.free(slices);
        try r.img.scatter(r.family.ptr, saved, slices, bytes);
        offset += count;
    }
    if (hash.final() != opened.trailer.payload_hash) return error.RowsHash;
}

test "streamed read windows preserve discontinuous image offsets across piece seams" {
    const a = std.testing.allocator;
    const ranges = [_]image.Range{ .{ .off = 0, .len = 8 }, .{ .off = 100, .len = 20 }, .{ .off = 300, .len = 16 } };
    const selected = try window(a, &ranges, 5, 30);
    defer a.free(selected);
    try std.testing.expectEqualSlices(image.Range, &.{ .{ .off = 5, .len = 3 }, .{ .off = 100, .len = 20 }, .{ .off = 300, .len = 7 } }, selected);
    try std.testing.expectError(error.RowsRead, window(a, &ranges, 43, 2));
}

test "streamed payload bounds staging and rejects a corrupt piece before the blank can be published" {
    const platform = @import("sys");
    const a = std.testing.allocator;
    var dir_buf: [96]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&dir_buf, "/tmp/tf-rows-stream-{d}", .{platform.c.getpid()}, 0);
    _ = platform.c.mkdir(dir, 0o700);
    defer _ = platform.c.rmdir(dir);
    const body = try a.alloc(u8, piece_bytes + 4096);
    defer a.free(body);
    for (body, 0..) |*byte, i| byte.* = @truncate(i *% 31 +% 7);
    var writer = try file.Writer.open(dir, "image", ".st");
    const name: @import("rows_chain.zig").Name = @splat(7);
    try writer.append(body);
    _ = try writer.finish(.{ .kind = @backingInt(file.Kind.state), .payload = 0, .payload_hash = 0, .name = name }, &.{}, false);
    var path_buf: [192]u8 = undefined;
    const path = try file.path(&path_buf, dir, "image", ".st", false);
    defer _ = platform.c.unlink(path);
    const Sink = struct {
        bytes: []u8,
        fn scatter(ptr: *anyopaque, _: pc.Saved, ranges: []const image.Range, in: []const u8) !void {
            const sink: *@This() = @ptrCast(@alignCast(ptr));
            var offset: usize = 0;
            for (ranges) |range| {
                const len: usize = @intCast(range.len);
                @memcpy(sink.bytes[@intCast(range.off)..][0..len], in[offset..][0..len]);
                offset += len;
            }
        }
    };
    var sink: Sink = .{ .bytes = try a.alloc(u8, body.len) };
    defer a.free(sink.bytes);
    const codec: @import("rows_store.zig").Image = .{ .streamed_reads = true, .parts = undefined, .gather = undefined, .blank = undefined, .scatter = Sink.scatter };
    var r: Rows = undefined;
    r.gpa = a;
    r.stage = &.{};
    r.img = &codec;
    r.family = .{ .ptr = &sink, .vtable = undefined };
    defer a.free(r.stage);
    var opened = try file.open(path, .state, name);
    defer opened.close();
    const ranges = [_]image.Range{.{ .off = 0, .len = body.len }};
    try payload(&r, &opened, &ranges, &sink);
    try std.testing.expectEqualSlices(u8, body, sink.bytes);
    try std.testing.expect(r.stage.len <= piece_bytes);
    const fd = platform.c.open(path, .{ .ACCMODE = .RDWR }, @as(platform.c.mode_t, 0));
    if (fd < 0) return error.RowsRead;
    defer _ = platform.c.close(fd);
    var flip = [_]u8{body[piece_bytes + 3] ^ 1};
    try std.testing.expectEqual(@as(isize, 1), platform.c.pwrite(fd, &flip, 1, @intCast(piece_bytes + 3)));
    try std.testing.expectError(error.RowsHash, payload(&r, &opened, &ranges, &sink));
}
