import re
import unicodedata

DIGITS = "零一二三四五六七八九"
UNITS = ((100_000_000, "亿"), (10_000, "万"), (1_000, "千"), (100, "百"), (10, "十"))


def chinese_integer(value, leading=True):
    if value < 10:
        return DIGITS[value]
    base, unit = next((base, unit) for base, unit in UNITS if value >= base)
    quotient, remainder = divmod(value, base)
    prefix = "" if leading and base == 10 and quotient == 1 else chinese_integer(quotient)
    if not remainder:
        return prefix + unit
    gap = "零" if remainder < base // 10 else ""
    return prefix + unit + gap + chinese_integer(remainder, leading=False)


def canonical(text):
    def spell_number(match):
        token = match.group()
        number = int(token.rstrip("%").replace(",", ""))
        return ("百分之" if token.endswith("%") else "") + chinese_integer(number)

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[0-9]+(?:,[0-9]{3})*%?", spell_number, text)
    return "".join(char for char in text if char.isalnum())


def distance(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  nxt=[i]
  for j,y in enumerate(b,1):nxt.append(min(prev[j]+1,nxt[-1]+1,prev[j-1]+(x!=y)))
  prev=nxt
 return prev[-1]
