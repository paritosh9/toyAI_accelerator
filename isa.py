# isa.py

from enum import Enum, auto

# =========================================================
# Supported opcodes for AI accelerator
# =========================================================
class OPCODES(Enum):
    # -------------------
    # Arithmetic (Elementwise Binary)
    # -------------------
    VADD = auto()
    VSUB = auto()
    VMUL = auto()
    VDIV = auto()

    # -------------------
    # Unary Math
    # -------------------
    VNEG = auto()
    VABS = auto()
    VSQRT = auto()
    VRSQRT = auto()
    VEXP = auto()
    VLOG = auto()

    # -------------------
    # Activations
    # -------------------
    RELU = auto()
    GELU = auto()
    SILU = auto()

    # -------------------
    # Reduction
    # -------------------
    REDUCE_SUM = auto()
    REDUCE_MEAN = auto()
    REDUCE_MAX = auto()

    # -------------------
    # Comparison / Logical
    # -------------------
    VCMP_EQ = auto()
    VCMP_NE = auto()
    VCMP_LT = auto()
    VCMP_LE = auto()
    VCMP_GT = auto()
    VCMP_GE = auto()
    VAND = auto()

    # -------------------
    # Layout / Tensor View
    # -------------------
    VIEW = auto()
    RESHAPE = auto()
    PERMUTE = auto()
    TRANSPOSE = auto()
    SLICE = auto()
    CONCAT = auto()
    EXPAND = auto()

    # -------------------
    # Tensor Creation
    # -------------------
    ARANGE = auto()
    ZEROS = auto()
    ONES = auto()
    FILL = auto()

    # -------------------
    # Linear Algebra
    # -------------------
    MATMUL = auto()
    BMM = auto()
    ADDMM = auto()

    # -------------------
    # Embedding
    # -------------------
    EMBED_LOOKUP = auto()

    # -------------------
    # Memory
    # -------------------
    DMA_LOAD = auto()
    DMA_STORE = auto()
    MEMORY_VIEW = auto()


# =========================================================
# Instruction Representation
# =========================================================
class Instruction:
    def __init__(self, instr_id, opcode, dst, src=None, meta=None):
        self.instr_id = instr_id
        self.opcode = opcode
        self.dst = dst
        self.src = src or []
        self.meta = meta

    def __str__(self):
        src_str = ", ".join(str(s) for s in self.src)
        meta_str = f", meta={self.meta}" if self.meta else ""
        return f"{self.instr_id}: {self.opcode.name} {self.dst}" + (
            f", {src_str}" if src_str else ""
        ) + meta_str

    def to_line(self):
        return str(self)


# =========================================================
# Program container (optional, used by AIProgram)
# =========================================================
class Program:
    def __init__(self):
        self.instructions = []

    def emit(self, instr_id, opcode, dst, src=None, meta=None):
        self.instructions.append(Instruction(instr_id, opcode, dst, src, meta))

    def dump(self):
        for instr in self.instructions:
            print(instr)


