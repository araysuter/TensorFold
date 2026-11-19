//! A snapshot's target and optional drafter bytes form one bounds-checked image.
const std = @import("std");

pub const Plane = struct { data: ?[]u8, len: usize };

pub fn transfer(first: []u8, second: []u8, ranges: anytype, buffer: []u8, read: bool) !void {
    return transferPlanes(&.{ .{ .data = first, .len = first.len }, .{ .data = second, .len = second.len } }, ranges, buffer, read);
}

pub fn transferPlanes(planes: []const Plane, ranges: anytype, buffer: []u8, read: bool) !void {
    var size: u64 = 0;
    for (planes) |plane| {
        if (plane.data) |bytes| if (bytes.len != plane.len) return error.InvalidSnapshotRange;
        size = std.math.add(u64, size, plane.len) catch return error.InvalidSnapshotRange;
    }
    var total: u64 = 0;
    for (ranges) |r| {
        const end = std.math.add(u64, r.off, r.len) catch return error.InvalidSnapshotRange;
        if (end > size) return error.InvalidSnapshotRange;
        total = std.math.add(u64, total, r.len) catch return error.InvalidSnapshotRange;
    }
    if (total != buffer.len) return error.InvalidSnapshotRange;
    var at: usize = 0;
    for (ranges) |r| {
        var off: usize = @intCast(r.off);
        var left: usize = @intCast(r.len);
        for (planes) |plane| {
            if (off >= plane.len) {
                off -= plane.len;
                continue;
            }
            const n = @min(left, plane.len - off);
            if (plane.data) |bytes| {
                if (read) @memcpy(buffer[at..][0..n], bytes[off..][0..n]) else @memcpy(bytes[off..][0..n], buffer[at..][0..n]);
            } else if (read) @memset(buffer[at..][0..n], 0);
            at += n;
            left -= n;
            off = 0;
            if (left == 0) break;
        }
    }
}

test "image copies across target and drafter boundaries and rejects invalid ranges before mutation" {
    const R = struct { off: u64, len: u64 };
    var first = [_]u8{ 1, 2, 3 };
    var second = [_]u8{ 4, 5 };
    var out: [4]u8 = undefined;
    try transfer(&first, &second, &[_]R{.{ .off = 1, .len = 4 }}, &out, true);
    try std.testing.expectEqualSlices(u8, &.{ 2, 3, 4, 5 }, &out);
    out = .{ 8, 7, 6, 5 };
    try transfer(&first, &second, &[_]R{.{ .off = 1, .len = 4 }}, &out, false);
    try std.testing.expectEqualSlices(u8, &.{ 1, 8, 7 }, &first);
    try std.testing.expectEqualSlices(u8, &.{ 6, 5 }, &second);
    try std.testing.expectError(error.InvalidSnapshotRange, transfer(&first, &second, &[_]R{.{ .off = 4, .len = 4 }}, &out, false));
    try std.testing.expectError(error.InvalidSnapshotRange, transfer(&first, &second, &[_]R{.{ .off = std.math.maxInt(u64), .len = 4 }}, &out, true));
    try std.testing.expectEqualSlices(u8, &.{ 1, 8, 7 }, &first);
}

test "a declined drafter's image is zero padded without exposing uninitialized bytes" {
    const R = struct { off: u64, len: u64 };
    var target = [_]u8{ 1, 2 };
    var out: [5]u8 = undefined;
    try transferPlanes(&.{ .{ .data = &target, .len = 2 }, .{ .data = null, .len = 3 } }, &[_]R{.{ .off = 0, .len = 5 }}, &out, true);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 0, 0, 0 }, &out);
}
