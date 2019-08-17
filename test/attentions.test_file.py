from src.modeling.seq2seq.Attentions import Attentions

methods = ["dot", "general", "concat"]

a = Attentions(methods[0], 500)
b = Attentions(methods[1], 500)
c = Attentions(methods[2], 500)

# Test Attentions with dot method
print("Attentions with dot method is ->", end=" ")
print(a)

# Test Attentions with general method
print("Attentions with general method is ->", end=" ")
print(b)

# Test Attentions with concat method
print("Attentions with concat method is ->", end=" ")
print(c)