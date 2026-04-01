from .parser import SimpleLangParser
from .vectorizer.vectorize import vectorize
from .vectorizer.dependency_testing import dependency_test
from ..assembly.lowerer import SimpleLangToAssembly
from ..assembly.codegen import SimpleLangAssembly2CCompiler
from ..assembly.codegen import SimpleLangAssembly2CGenerator

class SimpleLang2CCompiler:
    def __call__(self, prgm: str, vector_width: int, dump_c_code: bool = False):
        parsed_prgm = SimpleLangParser().parse(prgm)
        vectorized_prgm = vectorize(parsed_prgm, dependency_test)

        lowerer = SimpleLangToAssembly(vector_width)
        lowered_prgm = lowerer(vectorized_prgm)
        mod = SimpleLangAssembly2CCompiler()(lowered_prgm)

        if dump_c_code:
            c_code = SimpleLangAssembly2CGenerator()(lowered_prgm)
            with open("code.c", "w") as file:
                file.write(c_code)

        return mod