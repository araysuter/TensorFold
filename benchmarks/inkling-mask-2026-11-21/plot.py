"""Validate paired decode-mask receipts and render median throughput."""

import argparse
import json
import math
from pathlib import Path
import re
from statistics import median


HERE = Path(__file__).resolve().parent
LENGTHS = (2000, 8000, 32000, 64000)
MODES = ('baseline', 'mask')
REPEATS = 3
OUTPUT_TOKENS = 256


def read_receipt(path, expected_lengths):
    report = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(report, dict) or report.get('status') != 'passed':
        raise ValueError(f'{path.name} must be a complete, passed receipt')
    lengths = report.get('lengths', [])
    if sorted(lengths) != sorted(expected_lengths):
        raise ValueError(f'{path.name} has unexpected input lengths {lengths}')
    if report.get('repeats_per_length') != REPEATS or report.get('paired_repeats_verified') != REPEATS * len(lengths):
        raise ValueError(f'{path.name} must verify {REPEATS} pairs per input length')
    if report.get('output_tokens') != OUTPUT_TOKENS:
        raise ValueError(f'{path.name} must use {OUTPUT_TOKENS} output tokens')

    expected_keys = {(length, mode, repeat) for length in lengths for mode in MODES
                     for repeat in range(1, REPEATS + 1)}
    rows = {}
    for row in report.get('cells', []):
        key = row.get('input_tokens'), row.get('mode'), row.get('repeat')
        if key not in expected_keys or key in rows:
            raise ValueError(f'{path.name} has an unexpected or duplicate row {key}')
        if row.get('output_tokens') != OUTPUT_TOKENS or row.get('finish_reason') != 'length':
            raise ValueError(f'{path.name} did not complete the output budget for {key}')
        digest = row.get('token_sha256')
        if row.get('exact_to_first') is not True or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError(f'{path.name} did not verify the token hash for {key}')
        for metric in ('generation_tps', 'prompt_tps'):
            value = row.get(metric)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{path.name} has invalid {metric} for {key}')
        rows[key] = row
    if rows.keys() != expected_keys:
        raise ValueError(f'{path.name} is missing rows {sorted(expected_keys - rows.keys())}')
    for length in lengths:
        hashes = {row['token_sha256'] for key, row in rows.items() if key[0] == length}
        if len(hashes) != 1:
            raise ValueError(f'{path.name} has unequal token hashes at {length} input tokens')
    return report, rows


def render(rows, output_dir):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans',
                         'text.color': '#243247', 'axes.labelcolor': '#46546a',
                         'axes.titlecolor': '#243247', 'xtick.color': '#46546a',
                         'ytick.color': '#46546a', 'axes.edgecolor': '#b1bdd0'})
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    figure.subplots_adjust(left=.079, right=.97, top=.72, bottom=.20, wspace=.27)
    palette = ('#275f93', '#af4f24')
    labels = ('Baseline', 'Slice mask with BF16 select')
    settings = (('generation_tps', 'Decode', '{:.1f}'),
                ('prompt_tps', 'Prompt through first token', '{:,.0f}'))
    positions = list(range(len(LENGTHS)))
    width = .35
    for axis, (metric, title, number_format) in zip(axes, settings):
        samples = [[
            [rows[(length, mode, repeat)][metric] for repeat in range(1, REPEATS + 1)]
            for length in LENGTHS
        ] for mode in MODES]
        centers = [[median(values) for values in mode_samples] for mode_samples in samples]
        high = max(max(values) for mode_samples in samples for values in mode_samples)
        axis.set_ylim(0, 1.22 * high)
        for mode_index, (color, label) in enumerate(zip(palette, labels)):
            values = centers[mode_index]
            spread = samples[mode_index]
            errors = ([value - min(group) for value, group in zip(values, spread)],
                      [max(group) - value for value, group in zip(values, spread)])
            bar_positions = [x + (mode_index - .5) * (width + .045) for x in positions]
            axis.bar(bar_positions, values, width,
                     color=color, label=label, yerr=errors, zorder=3,
                     error_kw={'ecolor': '#243247', 'elinewidth': .9, 'capsize': 3, 'zorder': 4})
            for x, value in zip(bar_positions, values):
                axis.text(x, value * .88, number_format.format(value),
                          ha='center', va='top', color='white', fontsize=8.4, fontweight='bold', zorder=5)
        for x, baseline, mask in zip(positions, *centers):
            change = 100 * (mask / baseline - 1)
            extent = max(max(samples[0][x]), max(samples[1][x]))
            percent = '≈0.0%' if abs(change) < .05 else f'{change:+.1f}%'.replace('-', '−')
            axis.text(x, extent + .040 * high, percent,
                      ha='center', va='bottom', fontsize=10, fontweight='bold')
        axis.set_title(title, fontsize=13, pad=11, loc='left', fontweight='bold')
        axis.set_xlabel('Input tokens', labelpad=8)
        axis.set_ylabel('Tokens / second', labelpad=8)
        axis.set_xticks(positions, [f'{length // 1000}K' for length in LENGTHS])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        axis.grid(axis='y', color='#e2e8f0', linewidth=.8, zorder=0)
        axis.tick_params(length=0, pad=6)
        axis.spines[['top', 'right']].set_visible(False)

    figure.suptitle('Inkling 2 Mini decode mask on M5 Max', fontsize=20, fontweight='bold', y=.96)
    figure.text(.5, .872, '8-bit weights · three passes · 256 generated tokens per request',
                ha='center', fontsize=11, color='#46546a')
    figure.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .844),
                  ncol=2, frameon=False, fontsize=10)
    figure.text(.5, .060,
                'Median of three alternating pairs. Whiskers show observed min–max. Token hashes match within each input length.\n'
                'The separate 1,026-token cache seam check also matched and is omitted from the chart.',
                ha='center', va='center', fontsize=9, color='#46546a', linespacing=1.55)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / 'speed.svg', facecolor='white', metadata={'Date': None})
    svg = output_dir / 'speed.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    figure.savefig(output_dir / 'speed.png', dpi=160, facecolor='white')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eight-k', type=Path, default=HERE / 'mask-head-eight-k.json')
    parser.add_argument('--contexts', type=Path, default=HERE / 'mask-head-contexts.json')
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    try:
        eight_k, eight_k_rows = read_receipt(args.eight_k, (8000,))
        contexts, context_rows = read_receipt(args.contexts, (1026, 2000, 32000, 64000))
        for field in ('base_commit', 'head_commit', 'model_revision', 'model_config_sha256',
                      'worker_source_sha256', 'prefill_step_size', 'mlx', 'mlx_lm', 'mlx_env',
                      'weight_quantization_bits', 'loop_passes', 'metal_device', 'seed', 'temperature'):
            if field not in eight_k or eight_k[field] != contexts.get(field):
                raise ValueError(f'receipts disagree on {field}')
        if eight_k['metal_device'] != 'Apple M5 Max' or (eight_k['weight_quantization_bits'], eight_k['loop_passes']) != (8, 3):
            raise ValueError('receipts must use the fixed three-pass 8-bit Inkling model on M5 Max')
        render(eight_k_rows | context_rows, args.output_dir)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
