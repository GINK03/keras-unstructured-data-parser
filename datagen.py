
import msgpack
import random
import pickle
import glob
import sys
def datagen():
  terms = open("dataset/alice.txt").read().split()
  for i in range( 10000000 ): 
    iis = [ i for i in range(0,10000)]
    bbs = [ random.choice(iis) for i in range( random.randint(1, 20) ) ]
    jjs = [ random.choice(terms) for i in range( random.randint(1,10) ) ]
    unstructured = { "txt" : " ".join(jjs), "value" : bbs}
    s    = msgpack.packb( unstructured )
    raws = str( unstructured )
    mms  = str( s )
    if not ( len(raws) <= 200 and len(mms) <= 200 ) : 
      continue 
    eval( raws )
    tosave = [mms, raws]
    if i%500 == 0:
      print( i, tosave )
    open("dataset/%09d.pkl"%i, "wb").write( pickle.dumps(tosave) )

def build_dict():
  char_freq = {}
  for eg, name in enumerate( glob.glob("dataset/*.pkl") ):
    if eg%500 == 0:
      print( eg, name )
    mms, raws = pickle.loads( open(name, "rb").read() )
    for char in list( "".join( [mms, raws] ) ):
      if char_freq.get( char ) is None:
        char_freq[char] = len( char_freq )
  open("char_freq.pkl", "wb").write( pickle.dumps(char_freq) )

  char_index = {}
  for char, freq in char_freq.items():
    char_index[char] = len( char_index )
  open("char_index.pkl", "wb").write( pickle.dumps(char_index) )
if __name__ == '__main__':
  if '--step1' in sys.argv:
    datagen()

  if '--step2' in sys.argv:
    build_dict()
 
