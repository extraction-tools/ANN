import sys

if (len(sys.argv) < 2):
    quit()

filename = sys.argv[1]

f = open(filename, "r")

count = 0
for line in f:
    count += 1

f.close()

new_f = open("zeroes.dat","w")
for i in range(count-1):
    new_f.write("0\n")
new_f.write("0")

new_f.close()
