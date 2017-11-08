import sys,re
import nltk
from collections import defaultdict
import cfg_fix
from cfg_fix import parse_grammar, CFG
from pprint import pprint
# The printing and tracing functionality is in a separate file in order
#  to make this file easier to read
from cky_print import CKY_pprint, CKY_log, Cell__str__, Cell_str, Cell_log

class CKY:
    """An implementation of the Cocke-Kasami-Younger (bottom-up) CFG recogniser.

    Goes beyond strict CKY's insistance on Chomsky Normal Form.
    It allows arbitrary unary productions, not just NT->T
    ones, that is X -> Y with either Y -> A B or Y -> Z .
    It also allows mixed binary productions, that is NT -> NT T or -> T NT"""

    def __init__(self,grammar):
        '''Create an extended CKY processor for a particular grammar

        Grammar is an NLTK CFG
        consisting of unary and binary rules (no empty rules,
        no more than two symbols on the right-hand side

        (We use "symbol" throughout this code to refer to _either_ a string or
        an nltk.grammar.Nonterminal, that is, the two thinegs we find in
        nltk.grammar.Production)

        :type grammar: nltk.grammar.CFG, as fixed by cfg_fix
        :param grammar: A context-free grammar
        :return: none'''

        self.verbose=False
        assert(isinstance(grammar,CFG))
        self.grammar=grammar
        # split and index the grammar
        self.buildIndices(grammar.productions())

    def buildIndices(self,productions):
        '''
        args: list of productions with terminal and non-terminal components on lhs and rhs of every production.
            production - Each production maps a single symbol on the "left-hand side"(non-terminal) to a sequence of symbols on the "right-hand side"(terminals or non-terminal)
        
        Since it is a bottom-up approach, the rhs of the productions are made keys of the dictionary and lhs are the values. 
        This function segregates productions into unary and binary rules by checking the rhs length and returning them.

        returns: Two dictionaries those have unary and binary grammar rules.
        '''
        self.unary=defaultdict(list)
        self.binary=defaultdict(list)
        for production in productions:
            rhs=production.rhs()
            lhs=production.lhs()
            assert(len(rhs)>0 and len(rhs)<=2) # Cross-checking to CNF rule that states only 1 or 2 child(ren) are allowed 
            if len(rhs)==1:
                self.unary[rhs[0]].append(lhs)   # unary rules dictionary are filled
            else:
                self.binary[rhs].append(lhs)     # binary rules dictionary are filled

    def recognise(self,tokens,verbose=False):
        '''replace/expand this docstring. Your docs need NOT
        say anything more about the verbose option.

        Initialise a n * n+1 matrix to create a upper traingular matrix (or a parse traingle/chart) from the sentence,
        then run the CKY algorithm over it

        :type tokens:
        :param tokens:
        :type verbose: bool
        :param verbose: show debugging output if True, defaults to False
        :rtype: 
        :return:

        pre-condition illustration:
        #    1   2   3  ...
        # 0  .   .   .
        # 1      .   .
        # 2          .
        # ...

        post-condition illustration:
        #    1   2   3  ...
        # 0  X   A   S
        # 1      Y   B
        # 2          Z
        # ...

        where X, Y, Z are the words and A, B, S are non-terminals.

        '''
        self.verbose=verbose
        self.words = tokens
        self.n = len(self.words)+1    # +1 for readability despite zero indexing feature
        self.matrix = []
        # We index by row, then column
        #  So Y below is 1,2 and Z is 0,3
        #    1   2   3  ...
        # 0  .   .   Z
        # 1      Y   .
        # 2          .
        # ...
        for r in range(self.n-1):
             # rows
             row=[]
             for c in range(self.n): 
                 # columns, columns = row +1. This is done only in our implementation. Generally, it is 
                 # a square matrix where row = column. Here we are storing the words too. So words take up
                 # an extra column.
                 if c>r:
                     # This is one we care about, add a cell
                     row.append(Cell(r,c,self))
                 else:
                     # just a filler
                     row.append(None)
             self.matrix.append(row)
        self.unaryFill()
        self.binaryScan()
        # Replace the line below for Q6
        totalParsersNumber = len(self.matrix[0][self.n-1].labels())    # Get the number of labels in the top right corner of the tree. For the start of the sentence
        # print('------------------'+totalParsersNumber+'-----------------')
        if totalParsersNumber != 0:
            return totalParsersNumber
        else:
            return False   # In case the grammar doen't fit a sentence, you reach here.
    def unaryFill(self):
        '''
        args: none

        This method fills all the words in the bottom most cells of the parse tree 
        i.e., words are added on the diagonal of the upper triangular matrix.

        returns: none

        pre-condition illustration:
        #    1   2   3  ...
        # 0  .   .   .
        # 1      .   .
        # 2          .
        # ...

        post-condition illustration:
        #    1   2   3  ...
        # 0  X   .   .
        # 1      Y   .
        # 2          Z
        # ...
        
        where X, Y, Z are the words.

        '''
        for r in range(self.n-1):
            cell=self.matrix[r][r+1]
            word=self.words[r]
            cell.addLabel(word)    # Adds the word to the given cell.
            cell.unaryUpdate(word)  # Calls unaryUpdate, where addLabel() is called again. This will result in redundancy of labels.

    def binaryScan(self):
        '''(The heart of the implementation.)

Postcondition: the matrix has been filled with all constituents that
can be built from the input words and grammar.

How: Starting with constituents of length 2 (because length 1 has been
done already), proceed across the upper-right diagonals from left to
right and in increasing order of constituent length. Call maybeBuild
for each possible choice of (start, mid, end) positions to try to
build something at those positions.

        '''
        for span in range(2, self.n):
            for start in range(self.n-span):
                end = start + span
                for mid in range(start+1, end):
                    self.maybeBuild(start, mid, end)

    def maybeBuild(self, start, mid, end):
        '''
        start - commencing index of first cell. 
        end - ending of the first cell
        stop - ending of the second cell.
        
        Objective of this function is to compare two cells and check for rules from the binary rules. 

        Checks for every co-occurring adjacent terminals or non-terminals (rhs elements) in the binary dictionary and 
        returns the value of the rhs key from the binary dictionary. 
        Then it is passed back to the unaryUpdate function where the found elements are stacked 
        above the current co-occurring adjacent terminals or non-terminals in the matrix.

        '''
        self.log("%s--%s--%s:",start, mid, end)
        cell=self.matrix[start][end]
        for s1 in self.matrix[start][mid].labels():
            for s2 in self.matrix[mid][end].labels():
                if (s1,s2) in self.binary:
                    for s in self.binary[(s1,s2)]:
                        self.log("%s -> %s %s", s, s1, s2, indent=1)
                        cell.addLabel(s)    # Adds the word to the given cell.
                        cell.unaryUpdate(s,1)   # Calls unaryUpdate, where addLabel() is called again. This will result in redundancy of labels.

# helper methods from cky_print
CKY.pprint=CKY_pprint
CKY.log=CKY_log

class Cell:
    '''A cell in a CKY matrix'''
    def __init__(self,row,column,matrix):
        self._row=row
        self._column=column
        self.matrix=matrix
        self._labels=[]

    def addLabel(self,label):
        self._labels.append(label)

    def labels(self):
        return self._labels

    def unaryUpdate(self,symbol,depth=0,recursive=False):
        '''
        args: terminal (word from the sentence, if depth is 0) / non-terminals if depth is > 0

        Logs the terminal/non-terminal in the Cell. 
        Then checks for the terminal or non-terminal in the unary dictionary, 
        where terminal or non-terminal (patterns of interest) will be stored as the values with keys as the parent of the rule 
        (lhs in a production )
        '''
        if not recursive:
            self.log(str(symbol),indent=depth)
        if symbol in self.matrix.unary:
            for parent in self.matrix.unary[symbol]:
                self.matrix.log("%s -> %s",parent,symbol,indent=depth+1)
                self.addLabel(parent)   # Adds the label but unaryUpdate() below does it again.
                self.unaryUpdate(parent,depth+1,True)   # recursion enabled with depth incremented, which depicts the traversal is moving upwards 

# helper methods from cky_print
Cell.__str__=Cell__str__
Cell.str=Cell_str
Cell.log=Cell_log

class Label:
    '''A label for a substring in a CKY chart Cell

    Includes a terminal or non-terminal symbol, possibly other
    information.  Add more to this docstring when you start using this
    class'''
    def __init__(self,symbol,
                 # Fill in here, if more needed
                 ):
        '''Create a label from a symbol and ...
        :type symbol: a string (for terminals) or an nltk.grammar.Nonterminal
        :param symbol: a terminal or non-terminal
        '''
        self._symbol=symbol
        # augment as appropriate, with comments

    def __str__(self):
        return str(self._symbol)

    def __eq__(self,other):
        '''How to test for equality -- other must be a label,
        and symbols have to be equal'''
        assert isinstance(other,Label)
        return self._symbol==other._symbol

    def symbol(self):
        return self._symbol
    # Add more methods as required, with docstring and comments
