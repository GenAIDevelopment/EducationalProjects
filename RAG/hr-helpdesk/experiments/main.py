import os
def main():
    # print the process details like pid
    print("Started process with id: ", os.getpid())

    name = input("Enter your name: ")
    print("Hello from experiments!")


if __name__ == "__main__":
    main()
