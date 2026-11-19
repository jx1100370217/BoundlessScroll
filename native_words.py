import re
import unicodedata

def canonical(text):
 text=unicodedata.normalize('NFKC',text)
 text=text.replace('30%','百分之三十').replace('50,000,000','五千万').replace('5000万','五千万').replace('5000萬','五千萬').replace('20年','二十年')
 return ''.join(c for c in text if c.isalnum())
def distance(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  nxt=[i]
  for j,y in enumerate(b,1):nxt.append(min(prev[j]+1,nxt[-1]+1,prev[j-1]+(x!=y)))
  prev=nxt
 return prev[-1]
