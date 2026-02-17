# microcode.py
from isa import Instruction

class AIProgram:
    def __init__(self):
        self.instructions = []
        self._reg_map = {}
        self._reg_counter = 1

    def allocate_reg(self, x):
        if x not in self._reg_map:
            self._reg_map[x] = self._reg_counter
            self._reg_counter += 1
        return self._reg_map[x]

    def emit(self, instr_id, opcode, dst, src=None, meta=None):
        instr = Instruction(instr_id, opcode, dst, src, meta)
        self.instructions.append(instr)

    def dump_txt(self):
        return "\n".join(instr.to_line() for instr in self.instructions)

    def save_txt(self, filename):
        with open(filename, "w") as f:
            f.write(self.dump_txt())



