from autovec.simple_lang.compiler import SimpleLang2CCompiler

if __name__ == "__main__":
    with open("prgm.smpl", "r") as file:
        content = file.read()

    SimpleLang2CCompiler()(content, 8, True)