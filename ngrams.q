/N-grams in q
/////////////
/ 2015.01.06  - Version 1
/   - Known Issues:
/     - The n-gram counts are correct, but the probabilities math seems wrong;
/     - multigrams don't account for history/P_prior right;
/     - Needs backoff smoothing for unobserved n-grams;
/     - Should do a .u.upd pattern for adding counts, so peach and map+reduce are easy
/     - Corpus data needs scrubbed.  (e.g. delete non-word characters)
/   - Requires curl available on OS
/   - [MORE HERE]
/   - This is intended to show some basic patterns of q code that arise in natural language processing (NLP)
/////////////

/Set big IDE dimensions
\c 2000 1000
\C 2000 1000


/Download corpus.
ssrawin:system "curl http://norvig.com/ngrams/shakespeare.txt"

/Collapse lines & split on whitespace.
/This gives us 1 word/atom, as a string. Later, we'll index tables by n-gram, so symbols will be helpful.
sswords:" "vs""sv ssrawin
sssyms:`$sswords

unigrams:desc count each group sssyms
////Example usage:
/most+least likely unigrams:
/{[N] N#unigrams}   
/N<0 gives least likely unigrams.
/large N are candidates to ignore, since they're so common they provide little new information to the language model.
/{[N] N#unigrams}neg[1]  /gives words Shakespeare probably made up.

/Utility function for building probabilistic language model.
normalize:{x%sum x}
unigrammodel:0w^neg log normalize unigrams   /memoize this here

/
  Discussion:
Language models are often used to evaluate phrases in the ({neg log x};+) semiring (see Tropical Semiring)
In q, we must add (negative) logarithmic likelihoods of n-grams, instead of multiplying them.
Proof: For some N, we have exp[N]=0w.  0w + anything is 0w, rendering the language model unusable.

I find `unigrammodel (dictionary, defined above) a decent data structure for building & using language models:
 WARNINGS: Not tested at scale. I recommend using something like `history or `word to MAP, and (pj/) to REDUCE.
    +-> Full-scale language models have been found to cache poorly!    (by IBM speech group paper, [REFERENCE NEEDED])
    +-> i.e. there is not much overlap of word histories from t to (t+delta_t)  => human language is complex
    +-> 

The language model can tell us the relative likelihood of various phrases, so we can make statistical inferences with it.
q)sum 0w^neg[log normalize unigrams]  `the`dog`doctor
q)exp neg sum neg[log normalize unigrams]  `the`dog`doctor
  2.144713e-10          /We model the likelihood of any 3 words being "the dog doctor" at about 2 in 10 billion.
q)exp neg sum neg[log normalize unigrams]  `the`dog`house
  1.768294e-09          /For "the dog house", it's 1 in 1 billion.

So, "the dog house" is about 1-2 orders of magnitude more log-likely than "the dog doctor"
Let's generalize from here..

Note an n-gram is just a prior (n-1)-gram, with 1 more word appended.
We name the prior (n-1)-gram the `history, and the posterior appended unigram, `nextword.
 The nilgram is often called `epsilon in the natural language processing (NLP) literature.
 It is not against the rules to 'hallucinate' an epsilon, if that suits your needs. (warning:memory+speed penalty)
 Note, The likelihood of the `nilgram (0-gram) is 1f, else we abandon additive associativity.

Utility function for working with n-grams where n>1
  Note, unobserved n-grams will give infinity for the {neg log x}, or P=0.  This problem is often addressed with backoff/smoothing techniques.
\

multigrammodel:{[hist;nextword] sum unigrammodel hist,nextword}

/Then, example usages ...

/
q)`doctor`house!multigrammodel[`the`dog;] each `doctor`house
doctor| 22.26285
house | 20.15325

q)key unigrams
`,`the`and`to`I`of`you`a`my`;`in`is`not`that`me`.`it`be`with`your`!`his`for`:`this`have`him`he`will`thou`as`so`,And`but`her`thy`do`all`shall`are`thee`by`on`our`no`?`we`what`.I`good`from`at`am`lord`more`would`now`was`if`them`they`their`sir`love`man`she`or`us`To`here`come`hath`know`well`one`then`like`say`make`may`than`an`;And`upon`should`were`let`did`must`there`which`see`had`The`such`go`out`when`I'll`yet`too`,That`king`That`these`some`how`mine`can`give`take`up`speak`.What`.O`much`most`think`time`heart`never`tell`,To`'tis`life`men`death`art`great`hand`father`hear`,I`.And`very`doth`made`own`Of`any`O`again`.The`cannot`And`been`true`day`fair`where`away`done`,The`name`eyes`blood`As`.Why`leave`honour`world`sweet`.You`.My`thus`?I`But`son`noble`look`nor`before`old`ever`down`comes`fear`heaven`other`.But`way`night`pray`who`better`could`poor`Lord`What`into`nothing`hast`;For`God`being`A`.How`lady`myself`both`many`In`With`two`,As`word`,But`dead`brother`find`head`bear`little`live`.Come`every`;But`c..
q)count key unigrams
38782

Let's go for it..
Question: What words does Shakespeare write most often after "the dog" ?
q)key[unigrams]!multigrammodel[`the`dog;] each key[unigrams]

Above takes 30 seconds, and the result doesn't include any information about the history. 
  It's time to start thinking about using tables, and memoizing frequently seen intermediates... 
  So you can train your 2-gram from your 1-gram model in reasonable time, then your 3-gram from your 2-gram, and so on.


These are too slow to run on each reload. (and are now in a comment).
/unigrameval_thedog:key[unigrams]!multigrammodel[`the`dog;] each key[unigrams]
/unigrameval_thecat:key[unigrams]!multigrammodel[`the`cat;] each key[unigrams]
\

/Define unigram model evaluation table, which we'll build using the language model.
unigramevals:([] hist:(); word:`$(); p:`float$())

/Example usage:  
/insert[`unigramevals;(enlist`the`dog;`doctor; multigrammodel[`the`dog;`doctor])]    /commented, else duplicates

/Utility function to populate `modeltbl with likelihood of word `word appearing, given prior history of `hist
modelthw:{[modeltbl;hist;word]  insert[modeltbl;(enlist hist; word; multigrammodel[hist;word])]}

/Example usage:
modelthw[`unigramevals; `the`dog;] each `doctor`house`cat`sat`rat`stood`shall

/
Remember, p is a negative log likelihood.  Sort increasing to get decreasing likelihood.
q)`p xasc unigramevals
hist    word   p       
-----------------------
the dog shall  18.09856
the dog house  20.15325
the dog stood  21.72385
the dog doctor 22.26285
the dog cat    22.57115
the dog sat    22.78737
the dog rat    23.85208


Thoughts/notes for future work:
If parallelizing, we'd need to keep a separate copy of the unigram counts, and implement some strategy to update the probabilities table (like .u.upd in kdb+tick).
We'd need to store the reduced result as n-gram counts, before building the likelihood model.  `g# will speedup reads during model use.   pj/  is relevant for async accumulation of counted things.
Then we can accumulate observations as results come in from across the compute cluster we've run on.
\


/
Expected output:
q)\v
`ssrawin`sssyms`sswords`unigramevals`unigrammodel`unigrams
q)\f
`modelthw`multigrammodel`normalize
q)tables`.
,`unigramevals
\

/Some query-building around xprev does fast construction of trigrams.
`bigramhist`unigramhist`unigrams xcols -50# update trigrams:{x,y}'[bigramhist;unigrams]  from update bigramhist:{x,y}'[2 xprev unigrams;1 xprev unigrams], unigramhist:1 xprev unigrams from flip enlist[`unigrams]!enlist sssyms
trigramstbl:select trigrams from `bigramhist`unigramhist`unigrams xcols update trigrams:{x,y}'[bigramhist;unigrams]  from update bigramhist:{x,y}'[2 xprev unigrams;1 xprev unigrams], unigramhist:1 xprev unigrams from flip enlist[`unigrams]!enlist sssyms  /runtime ~200 ms

/
Example use:
q)\t desc count each group trigramstbl
470

q)100#asc neg log normalize count each group trigramstbl
trigrams         |         
-----------------| --------
,    my     lord | 6.880817
,    sir    ,    | 7.107727
my   lord   ,    | 7.882265
,    I      will | 7.933251
,    I      am   | 8.111274
,    and    the  | 8.203647
,    and    I    | 8.346438
;    and    ,    | 8.374742
,    as     I    | 8.459581
\


/
References:
 - http://en.wikipedia.org/wiki/Semiring
 - [MORE HERE]

\
