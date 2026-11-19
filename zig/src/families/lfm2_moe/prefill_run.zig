//! One prompt chunk with the provider's owned drafter taps and kept-state marks.
const lanes = @import("lanes");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const be = lanes.backend;

pub fn For(comptime Metal: type) type {
    return struct {
        m: *Metal,
        cache: *st.Cache,
        sink: ?be.FeatureSink,
        later: bool,

        pub fn run(r: @This(), stream: *lanes.Stream, at: usize, end: usize) !void {
            const self = r.m;
            const ids = stream.prompt();
            var wait: ?fwd.Wait = null;
            const tap = try self.tapFor(stream, at, end - at, &wait);
            if ((r.sink != null or r.later) and tap == .none) return error.LfmTapsNotHeld;
            try self.run(&.{.{ .cache = r.cache, .rows = end - at, .tap = tap }}, ids[at..end], false, .last, wait);
            if (r.sink != null or r.later) {
                r.cache.tap = self.tapped(r.cache, end - at, tap, 0);
                if (r.later) r.cache.chunk_tap = r.cache.tap;
                if (r.sink) |k| {
                    const t = self.scratch.taps.?;
                    const f = try Metal.featuresFn(self, stream, t.layers[0..t.count], at, @intCast(end - at));
                    try k.take(k.ptr, stream, f, at);
                }
                if (end == ids.len) {
                    if (tap == .stage) {
                        r.cache.tap = .{ .start = end - 1, .row = 0, .rows = 1, .gen = 0, .stage = tap.stage.off + (end - at - 1) * self.tapBytes() };
                    } else try self.keepLastTap(r.cache, end - 1 - at, end - 1);
                }
            }
            lanes.prefill_widths.completed(self.gpa, stream, end - at);
        }

        pub fn keep(_: @This(), stream: *lanes.Stream, at: u32) void {
            if (stream.reuse.hook) |hk| hk.at(hk.ptr, stream, at);
        }
    };
}
