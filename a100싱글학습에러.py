edited_lines = []

YOUR_DIR = '/usr/local/lib/python3.8/dist-packages/torch/cuda/amp/grad_scaler.py'

with open(YOUR_DIR) as f:
    lines = f.readlines()
    for line in lines:
        # 조건에 따라 원하는 대로 line을 수정
        if 'optimizer_state["found_inf_per_device"] = self._unscale_grads_(optimizer, inv_scale, found_inf, False)' in line:
            edited_lines.append(line.replace("False", "True"))
        else:
            edited_lines.append(line)

with open(YOUR_DIR, 'w') as f:
    f.writelines(edited_lines)