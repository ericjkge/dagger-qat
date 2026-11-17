import sys
p = sys.argv[1]
s = open(p + ".orig").read()
old = '        """Record one step accepted draft count (excludes bonus token) into the histogram."""\n'
new = old + (
    '        _p = __import__("os").environ.get("PEAR_SPEC_STEP_LOG")\n'
    '        if _p:\n'
    '            with open(_p, "a") as _f:\n'
    '                _f.write(f"{self.rid} {len(self.output_ids)} {int(num_correct_drafts)}\\n")\n'
)
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
