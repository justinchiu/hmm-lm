header = "#!/bin/bash"
print(header)

def make_fn_str(model, bsz, dim, k=None):
    decl = f"{model}_b{bsz}_d{dim}"
    if k is not None:
        decl += f"_k{k}"
    config = f"{model}-d{dim}"
    if k is not None:
        config += f"-k{k}"
    log = f"{model}_b{bsz}_d{dim}"
    if k is not None:
        log += f"_k{k}"
    fn_str = f"""function {decl} {{
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \\
        --model_config configs/{config}.yaml \\
        --bsz {bsz} \\
        --num_epochs 100 \\
        --patience 8 \\
        --save {decl} \\
        > logs/{log}.log
}}"""
    return fn_str


print("# Lstm sweep")
bszs = [4, 32]
dims = [256, 512, 650]
model = "lstm"
for bsz in bszs:
    for dim in dims:
        print(make_fn_str(model, bsz, dim))

print("# Ff sweep")
bszs = [4, 32]
dims = [256, 512]
ks = [2, 3, 4, 5]
model = "ff"
for bsz in bszs:
    for dim in dims:
        for k in ks:
            print(make_fn_str(model, bsz, dim, k))


print("# Hmm sweep")
model = "hmm"
bszs = [4, 32]
dims = [256, 512]
ks = [128, 256]
for bsz in bszs:
    for dim in dims:
        for k in ks:
            print(make_fn_str(model, bsz, dim, k))

# overfitting script
def make_fn_str_overfit(model, bsz, dim, k=None):
    decl = f"{model}_b{bsz}_d{dim}"
    if k is not None:
        decl += f"_k{k}"
    decl += "_overfit"
    config = f"{model}-d{dim}"
    if k is not None:
        config += f"-k{k}"
    log = f"{model}_b{bsz}_d{dim}"
    if k is not None:
        log += f"_k{k}"
    log += "_overfit"
    fn_str = f"""function {decl} {{
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \\
        --model_config configs/{config}.yaml \\
        --num_epochs 100 \\
        --bsz {bsz} \\
        --patience 2 \\
        --overfit \\
        > overfit_logs/{log}.log
}}"""
    return fn_str

print("# Hmm overfitting script")
bsz = 4
model = "hmm"
dims = [256, 512]
ks = [128, 256, 512]

for dim in dims:
    for k in ks:
        print(make_fn_str_overfit(model, bsz, dim, k))

# tvm script
def make_fn_str_tvm(model, bsz, dim, k=None):
    decl = f"{model}_b{bsz}_d{dim}"
    if k is not None:
        decl += f"_k{k}"
    decl += "_tvm"
    config = f"{model}-d{dim}"
    if k is not None:
        config += f"-k{k}"
    log = f"{model}_b{bsz}_d{dim}"
    if k is not None:
        log += f"_k{k}"
    log += "_tvm"
    fn_str = f"""function {decl} {{
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \\
        --model_config configs/{config}.yaml \\
        --num_epochs 100 \\
        --bsz {bsz} \\
        --patience 8 \\
        --save {decl} \\
        > tvm_logs/{log}.log
}}"""
    return fn_str

print("# Hmm tvm script")
bsz = 4
model = "hmm"
dims = [256, 512]
ks = [128, 256, 512, 1024]

for dim in dims:
    for k in ks:
        print(make_fn_str_tvm(model, bsz, dim, k))

print("# Hmm seed sweep")
def make_fn_str_hmm_seed(model, bsz, dim, k, seed):
    decl = f"{model}_b{bsz}_d{dim}_k{k}_s{seed}"
    config = f"{model}-d{dim}-k{k}"
    log = f"{model}_b{bsz}_d{dim}_k{k}_s{seed}"
    fn_str = f"""function {decl} {{
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \\
        --model_config configs/{config}.yaml \\
        --bsz {bsz} \\
        --num_epochs 100 \\
        --patience 8 \\
        --save {decl} \\
        --seed {seed} \\
        > logs/{log}.log
}}"""
    return fn_str

model = "hmm"
bszs = [4]
dims = [256]
ks = [512]
seeds = [1234, 1, 2357, 2468]
for bsz in bszs:
    for dim in dims:
        for k in ks:
            for seed in seeds:
                print(make_fn_str_hmm_seed(model, bsz, dim, k, seed))

# tvm and old resnet script (best so far)
def make_fn_str_oldres(model, bsz, dim, k):
    decl = f"{model}_b{bsz}_d{dim}_k{k}"
    decl += "_oldres"
    config = f"{model}-d{dim}-k{k}-oldres"
    log = f"{model}_b{bsz}_d{dim}_k{k}_oldres"
    fn_str = f"""function {decl} {{
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \\
        --model_config configs/{config}.yaml \\
        --num_epochs 100 \\
        --bsz {bsz} \\
        --patience 8 \\
        --save {decl} \\
        > tvm_logs/{log}.log
}}"""
    return fn_str

print("# Hmm tvm oldres script")
bsz = 4
model = "hmm"
dims = [256, 512]
ks = [128, 256, 512, 1024]

for dim in dims:
    for k in ks:
        print(make_fn_str_oldres(model, bsz, dim, k))

# tvm and old resnet script (best so far)
def make_fn_str_poe_oldres(model, bsz, dim, n, k):
    decl = f"{model}_b{bsz}_d{dim}_n{n}_k{k}"
    decl += "_oldres"
    config = f"{model}-d{dim}-n{n}-k{k}-oldres"
    log = f"{model}_b{bsz}_d{dim}_n{n}_k{k}_oldres"
    fn_str = f"""function {decl} {{
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \\
        --model_config configs/{config}.yaml \\
        --num_epochs 100 \\
        --bsz {bsz} \\
        --patience 8 \\
        --save {decl} \\
        > tvm_logs/{log}.log
}}"""
    return fn_str

print("# PoeHmm tvm oldres script")
bsz = 4
model = "poehmm"
dims = [256, 512]
ns = [2, 3, 4, 5]
ks = [128, 256, 512, 1024]

for dim in dims:
    for n in ns:
        for k in ks:
            print(make_fn_str_poe_oldres(model, bsz, dim, n, k))

