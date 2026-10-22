"""Adapt the pinned Ornith native port to independently quantized shared experts.

Applies only to BobClawblaw/TensorFold 9328aff8a6c470b026aee7cefe53a0527de44814.
No checkpoint tensors or quantization parameters are changed.
"""
from pathlib import Path
import sys


def replace_once(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1, (str(path), text.count(old))
    path.write_text(text.replace(old, new))


def main(root):
    # Scratch scales with prompt-frame rows; 128 rows allocates several GiB of
    # recurrent snapshots beside the weights. 32 still fits a 16-row draft window.
    replace_once(root / 'zig/src/native/qwen27_served.zig',
        'q.model.Model.load(gpa, io, dir, 128, shareFits)',
        'q.model.Model.load(gpa, io, dir, 32, shareFits)')
    host = root / 'zig/src/core/row_projection.zig'
    replace_once(host,
        'extra.group != one.group or extra.bits != one.bits or extra.sum != one.sum',
        'extra.sum != one.sum')
    shader = root / 'zig/kernels/metal/core/row_projection.metal'
    text = shader.read_text()
    start = text.index('[[kernel]] void tf_row_projection_indexed_extra(')
    end = text.index('\ntemplate [[host_name', start)
    block = text[start:end]
    block = block.replace('const device bfloat* B2 [[buffer(9)]],',
        'const device bfloat* B2 [[buffer(9)]],\n'
        '                                          constant int* extra_dims [[buffer(10)]],')
    old = '''  const device uint8_t* wr = (const device uint8_t*)(extra ? W2 : W) + at * (K * BITS / 8);
  const device bfloat* sr = (extra ? S2 : S) + at * (K / GS);
  const device bfloat* br = (extra ? B2 : B) + at * (K / GS);
  float acc[1][RPS];
  tf_rp_rows<RPS, 1, BITS, BF16SUM>(wr, sr, br, X + size_t(slot / repeat) * K, K, GS, lane, acc);'''
    cases = '\n'.join(
        f'      case {bits}: tf_rp_rows<RPS, 1, {bits}, BF16SUM>(wr, sr, br, X + size_t(slot / repeat) * K, K, group, lane, acc); break;'
        for bits in [2, 3, 4, 5, 6, 8])
    new = '''  const int bits = extra ? extra_dims[0] : BITS;
  const int group = extra ? extra_dims[1] : GS;
  const device uint8_t* wr = (const device uint8_t*)(extra ? W2 : W) + at * (K * bits / 8);
  const device bfloat* sr = (extra ? S2 : S) + at * (K / group);
  const device bfloat* br = (extra ? B2 : B) + at * (K / group);
  float acc[1][RPS];
  if (extra) {
    switch (bits) {
''' + cases + '''
    }
  } else {
    tf_rp_rows<RPS, 1, BITS, BF16SUM>(wr, sr, br, X + size_t(slot / repeat) * K, K, GS, lane, acc);
  }'''
    assert block.count(old) == 1
    block = block.replace(old, new)
    shader.write_text(text[:start] + block + text[end:])
    replace_once(root / 'zig/src/families/qwen27/moe_gpu.zig',
        '        e.setBuffer(extra.biases, extra.b_off, 9);',
        '        e.setBuffer(extra.biases, extra.b_off, 9);\n'
        '        e.setValue([2]i32{ @intCast(extra.bits), @intCast(extra.group) }, 10);')
    test = root / 'zig/tests/row_projection_check.zig'
    replace_once(test, '    var checked: usize = 0;',
        '    try mixedSharedCheck(device, queue, &p);\n    var checked: usize = 0;')
    with test.open('a') as f:
        f.write(MIXED_TEST)


MIXED_TEST = r'''
const IndexedExtra = struct {
    p: *const row.Pipelines, x: row.Experts, extra: row.Weights,
    in: mtl.Buffer, slots: usize, ids: mtl.Buffer, out: mtl.Buffer,
    fn encode(d: @This(), e: mtl.ComputeEncoder) !void {
        const c = try row.callIndexedExtra(d.p, d.x, d.extra, d.slots);
        e.setPipeline(c.pipeline);
        e.setBuffer(d.in, 0, 0);
        e.setBuffer(d.x.one.w, 0, 1);
        e.setBuffer(d.x.one.scales, 0, 2);
        e.setBuffer(d.x.one.biases, 0, 3);
        e.setValue(c.dims, 4);
        e.setBuffer(d.out, 0, 5);
        e.setBuffer(d.ids, 0, 6);
        e.setBuffer(d.extra.w, 0, 7);
        e.setBuffer(d.extra.scales, 0, 8);
        e.setBuffer(d.extra.biases, 0, 9);
        e.setValue([2]i32{ @intCast(d.extra.bits), @intCast(d.extra.group) }, 10);
        e.dispatchGroups(mtl.Size.of(c.groups, c.slots, 1), mtl.Size.of(c.threads, 1, 1));
    }
};

fn mixedSharedCheck(device: mtl.Device, queue: mtl.Queue, p: *const row.Pipelines) !void {
    for ([_]u8{ 2, 3, 4 }) |width| for ([_]usize{ 32, 64 }) |group| {
        const c = Case{ .n = 64, .k = 128, .experts = 3, .rows = 6, .repeat = 3, .bits = width };
        const s = try Synth.init(device, c, 41);
        defer s.deinit();
        const ec = Case{ .n = 64, .k = 128, .experts = 1, .rows = 6, .repeat = 1, .bits = 4 };
        const extra = try Synth.init(device, ec, 91);
        defer extra.deinit();
        // Fixture has group-64-sized metadata; group 32 needs twice as much.
        const scales = try device.buffer(c.experts * c.n * (c.k / group) * 2, opts);
        defer scales.deinit();
        const biases = try device.buffer(c.experts * c.n * (c.k / group) * 2, opts);
        defer biases.deinit();
        for (scales.slice(u16, c.experts * c.n * (c.k / group))) |*v| v.* = toBf(0.013);
        for (biases.slice(u16, c.experts * c.n * (c.k / group))) |*v| v.* = toBf(-0.021);
        const one = row.Weights{ .w = s.w, .scales = scales, .biases = biases,
            .n = c.n, .k = c.k, .bits = width, .group = group };
        const ew = row.Weights{ .w = extra.w, .scales = extra.s, .biases = extra.b,
            .n = c.n, .k = c.k, .bits = 4, .group = 64 };
        const ids = s.ids.slice(u32, 6);
        @memcpy(ids, &[_]u32{ 0, 3, 1, 3, 2, 4 });
        try run(queue, IndexedExtra{ .p = p, .x = .{ .one = one, .experts = 3, .repeat = 3 },
            .extra = ew, .in = s.x, .slots = 6, .ids = s.ids, .out = s.out });
        const got = try a.dupe(u16, s.out.slice(u16, 6 * c.n));
        defer a.free(got);
        for (0..6) |slot| {
            const id = ids[slot];
            if (id > 3) {
                for (got[slot*c.n..][0..c.n]) |v| if (v != 0) return error.MissingExpertNotZero;
                continue;
            }
            var w = if (id == 3) ew else one;
            if (id != 3) {
                w.w_off = id * c.n * c.k * width / 8;
                w.s_off = id * c.n * (c.k / group) * 2;
                w.b_off = w.s_off;
            }
            const input = try device.buffer(c.k * 2, opts);
            defer input.deinit();
            @memcpy(input.slice(u16, c.k), s.x.slice(u16, c.rows * c.k)[(slot/3)*c.k..][0..c.k]);
            try run(queue, Dense{ .p = p, .w = w, .x = input, .rows = 1, .out = s.one, .relu2 = false });
            for (got[slot*c.n..][0..c.n], s.one.slice(u16, c.n)) |g, want|
                if (g != want) return error.MixedSharedDiffersFromDense;
        }
        std.debug.print("PASS mixed q{d} group {d} experts + q4 group64 shared, repeat3, missing-id zero\n", .{ width, group });
    };
}
'''


if __name__ == '__main__':
    main(Path(sys.argv[1]))
