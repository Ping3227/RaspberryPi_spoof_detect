import os
path = "../"
for response in os.walk(path):
    print(response)

# for root , dir ,file in os.walk(path):
#     for f in file:
#         print(root)
#         print(os.path.join(root,f))
