from johnim.util import *


class SdScript:
    def __init__(self, raw_lines=None, filename=None):
        if raw_lines is None and filename is None:
            raise Exception("Can't initialize SdScript from nothing!")
        
        if raw_lines is not None:
            self.raw_lines = raw_lines
            self.source_file = filename
        else:
            self.source_file = filename
            with open(filename, 'r') as f:
                self.raw_lines = f.readlines()
        
        self.processed = preprocess([x.strip() for x in self.raw_lines])

    def rewrite(self, raw_lines):
        self.raw_lines = raw_lines

    def expand(self):
        self.raw_lines = [x + '\n' for x in self.processed]

    def save(self, new_filename=None):
        if new_filename is not None:
            self.source_file = new_filename
        if self.source_file is None:
            raise Exception("No file name")
        
        with open(self.source_file, 'w') as f:
            f.writelines(self.raw_lines)



testo = SdScript(filename='sdscripts/testomatic.sd')

print(testo.source_file)
print('=================')
print(testo.raw_lines)
print('=================')
print(testo.processed)

testo.expand()

testo.save(new_filename='sdscripts/testomatic2.sd')