from src.classes.BagsOfWords import BagsOfWords as WordBag
import src.processing.vector_matrix as matrix_operater

bag = WordBag()

a = [["EOS"], ["PAD", "EOS"]]
b = []

print("First Data is ", end="")
print(a)

# Test encoded_string working
for aa in a:
    b.append(
        matrix_operater.encoded_string(aa, bag)
    )
print("After encoded is ", end="")
print(b)

# Test decoded_string working
c = []

for bb in b:
    c.append(
        matrix_operater.decoded_string(bb, bag)
    )

print("After decoded is ", end="")
print(c)

# Test counting working
for bb in b:
    print("count ", end="")
    print(bb, end=" ")
    print("excluded '0' gets result as " + \
            str(matrix_operater.counting(bb, 0))
        )

# Test binary_matrix woring
for cc in c:
    print("binary of ", end="")
    print(cc, end=" ")
    print("excluded 'PAD' gets results as ", end="")
    print(matrix_operater.binary_matrix(cc, bag, "PAD"))

