# lowering.py
from isa import OPCODES
from microcode import AIProgram

def lower_fx_graph(graph, program: AIProgram):
    instr_id = 0
    reg_map = {}
    next_reg = 0

    # ----------------------------------------------
    # Simple register allocator
    # ----------------------------------------------
    def get_reg(name):
        nonlocal next_reg
        if name not in reg_map:
            reg_map[name] = next_reg
            next_reg += 1
        return reg_map[name]

    # ----------------------------------------------
    # Walk FX Graph
    # ----------------------------------------------
    for node in graph.graph.nodes:

        # ------------------------------
        # Input placeholder
        # ------------------------------
        if node.op == "placeholder":
            dst = get_reg(node.name)
            program.emit(instr_id, OPCODES.DMA_LOAD, dst, [])
            instr_id += 1
            continue

        # ------------------------------
        # call_function
        # ------------------------------
        if node.op == "call_function":
            target = str(node.target)
            dst = get_reg(node.name)
            src = [get_reg(arg.name) for arg in node.args if hasattr(arg, "name")]

            # Skip harmless helpers
            if any(x in target for x in ["sym_size", "_set_grad_enabled", "to", "float", "new_ones"]):
                continue

            # -------- Arithmetic --------
            if "add" in target:
                opcode = OPCODES.VADD
            elif "sub" in target:
                opcode = OPCODES.VSUB
            elif "mul" in target:
                opcode = OPCODES.VMUL
            elif "div" in target:
                opcode = OPCODES.VDIV
            # -------- Unary --------
            elif "rsqrt" in target:
                opcode = OPCODES.VRSQRT
            elif "sqrt" in target:
                opcode = OPCODES.VSQRT
            elif "exp" in target:
                opcode = OPCODES.VEXP
            elif "log" in target:
                opcode = OPCODES.VLOG
            elif "neg" in target:
                opcode = OPCODES.VNEG
            elif "abs" in target:
                opcode = OPCODES.VABS
            # -------- Activations --------
            elif "silu" in target:
                opcode = OPCODES.SILU
            elif "gelu" in target:
                opcode = OPCODES.GELU
            elif "relu" in target:
                opcode = OPCODES.RELU
            # -------- Reduction --------
            elif "mean" in target:
                opcode = OPCODES.REDUCE_MEAN
            elif "sum" in target:
                opcode = OPCODES.REDUCE_SUM
            elif "amax" in target:
                opcode = OPCODES.REDUCE_MAX
            # -------- Layout --------
            elif "getitem" in target:
                opcode = OPCODES.SLICE
            elif "cat" in target:
                opcode = OPCODES.CONCAT
            elif "permute" in target:
                opcode = OPCODES.PERMUTE
            elif "transpose" in target:
                opcode = OPCODES.TRANSPOSE
            elif "view" in target:
                opcode = OPCODES.VIEW
            # -------- Linear Algebra / Embedding --------
            elif "embedding" in target:
                opcode = OPCODES.EMBED_LOOKUP
            else:
                # Skip everything else like pow, cos, sin, cumsum
                continue

            program.emit(instr_id, opcode, dst, src)
            instr_id += 1
            continue

        # ------------------------------
        # call_method
        # ------------------------------
        if node.op == "call_method":
            method = node.target
            dst = get_reg(node.name)
            src = [get_reg(arg.name) for arg in node.args if hasattr(arg, "name")]

            # Skip harmless helpers
            if method in ["unsqueeze", "to", "float", "contiguous", "new_ones", "cumsum"]:
                continue

            # Layout / tensor view ops
            if method == "view":
                opcode = OPCODES.VIEW
            elif method == "reshape":
                opcode = OPCODES.RESHAPE
            elif method == "permute":
                opcode = OPCODES.PERMUTE
            elif method == "transpose":
                opcode = OPCODES.TRANSPOSE
            elif method == "expand":
                opcode = OPCODES.EXPAND
            else:
                continue

            program.emit(instr_id, opcode, dst, src)
            instr_id += 1
            continue

        # ------------------------------
        # call_module
        # ------------------------------
        if node.op == "call_module":
            module = graph.owning_module.get_submodule(node.target)
            module_name = module.__class__.__name__

            dst = get_reg(node.name)
            src = [get_reg(arg.name) for arg in node.args if hasattr(arg, "name")]

            if module_name == "Linear":
                opcode = OPCODES.MATMUL
            elif module_name == "Embedding":
                opcode = OPCODES.EMBED_LOOKUP
            else:
                continue

            program.emit(instr_id, opcode, dst, src)
            instr_id += 1
            continue

    return program







