
import math
import time

def euclid(a, b):
    x0, x1, y0, y1 = 1, 0, 0, 1

    while b != 0:
        q = a // b
        r = a % b
        print(f"{a} = {q} * {b} + {r}")
        a, b = b, r

        tempX = x0
        x0, x1 = x1, tempX - q * x1

        tempY = y0
        y0, y1 = y1, tempY - q * y1

    return a, x0, y0

def sieve(a, b):
    is_prime = [True] * (b + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(math.sqrt(b)) + 1):
        if is_prime[i]:
            for j in range(i * i, b + 1, i):
                is_prime[j] = False

    primes = [i for i in range(a, b + 1) if is_prime[i]]
    return primes

def factorization(n):
    prime = sieve(0, 1000000)
    p, e = [], []
    current = n
    i = 0

    while i < len(prime):
        if current % prime[i] == 0:
            if prime[i] in p:
                index = p.index(prime[i])
                e[index] += 1
            else:
                p.append(prime[i])
                e.append(1)
            current //= prime[i]
        else:
            i += 1
            temp = prime[i] * prime[i]
            if temp > current:
                break

    if current > 1:
        p.append(current)
        e.append(1)

    return p, e

def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)

def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1

    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

def lineareqsolver(a, b, c):
    g = gcd(abs(a), abs(b))
    if c % g != 0:
        print("No integer solutions.")
        return
    a, b = abs(a), abs(b)
    g_extended, x0, y0 = extended_gcd(a, b)

    x0 *= c // g
    y0 *= c // g

    print(f"Initial solution: x = {x0}, y = {y0}")
    print("Other Solutions:")
    print(f"x = {b // g}t {'+ ' if (x0 > 0) else '- '}{abs(x0)}")
    print(f"y = {-a // g}t {'+ ' if (y0 > 0) else '- '}{abs(y0)}")

def multinv(a, m):
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        print("Inverse doesn't exist (a and m are not coprime).")
        return -1

    result = (x % m + m) % m
    return result

def modexp(a, x, m):
    result = 1
    a = a % m

    while x > 0:
        if x % 2 == 1:
            result = (result * a) % m
        x = x // 2
        a = (a * a) % m

    return result

# A5
def chinrem(moduli, remainders):
    M = 1
    for m in moduli:
        M *= m

    result = 0
    for i in range(len(moduli)):
        Mi = M // moduli[i]
        result += remainders[i] * Mi * pow(Mi, -1, moduli[i])

    return result % M

def caesar_cipher(text, shift):
    encrypted_text = ""
    for char in text:
        if char.isalpha():
            is_upper = char.isupper()
            char = char.lower()
            shifted = chr(((ord(char) - ord('a') + shift) % 26) + ord('a'))
            if is_upper:
                shifted = shifted.upper()
            encrypted_text += shifted
        else:
            encrypted_text += char
    return encrypted_text

def caesar_decipher(text, shift):
    return caesar_cipher(text, -shift)

def affine_cipher(text, a, b):
    encrypted_text = ""
    for char in text:
        if char.isalpha():
            is_upper = char.isupper()
            char = char.lower()
            shifted = chr(((a * (ord(char) - ord('a')) + b) % 26) + ord('a'))
            if is_upper:
                shifted = shifted.upper()
            encrypted_text += shifted
        else:
            encrypted_text += char
    return encrypted_text

def affine_decipher(text, a, b):
    a_inv = None
    for i in range(26):
        if (a * i) % 26 == 1:
            a_inv = i
            break
    if a_inv is not None:
        b_inv = (-a_inv * b) % 26
        return affine_cipher(text, a_inv, b_inv)
    else:
        return "Invalid 'a' value, no modular multiplicative inverse exists."


char_to_num = {
    'a': 11, 'b': 12, 'c': 13, 'd': 14, 'e': 15,
    'f': 16, 'g': 17, 'h': 18, 'i': 19, 'j': 20,
    'k': 21, 'l': 22, 'm': 23, 'n': 24, 'o': 25,
    'p': 26, 'q': 27, 'r': 28, 's': 29, 't': 30,
    'u': 31, 'v': 32, 'w': 33, 'x': 34, 'y': 35, 'z': 36,
    'A': 37, 'B': 38, 'C': 39, 'D': 40, 'E': 41,
    'F': 42, 'G': 43, 'H': 44, 'I': 45, 'J': 46,
    'K': 47, 'L': 48, 'M': 49, 'N': 50, 'O': 51,
    'P': 52, 'Q': 53, 'R': 54, 'S': 55, 'T': 56,
    'U': 57, 'V': 58, 'W': 59, 'X': 60, 'Y': 61, 'Z': 62,
    '0': 63, '1': 64, '2': 65, '3': 66, '4': 67,
    '5': 68, '6': 69, '7': 70, '8': 71, '9': 72,
    '.': 73, ',': 74, '!': 75, '?': 76, ':': 77,
    ';': 78, '=': 79, '+': 80, '-': 81, '*': 82,
    '/': 83, '^': 84, '\\': 85, '@': 86, '#': 87,
    '&': 88, '(': 89, ')': 90, '[': 91, ']': 92,
    '{': 93, '}': 94, '$': 95, '%': 96, '_': 97,
    '`': 98, ' ': 99
}

num_to_char = {v: k for k, v in char_to_num.items()}

def text_to_numeric(message):
    numeric_chunks = []
    for char in message:
        if char in char_to_num:
            numeric_chunks.append(str(char_to_num[char]))
    return numeric_chunks

def numeric_to_text(numeric_chunks):
    text = ""
    for chunk in numeric_chunks:
        num = int(chunk)
        if num >= 11 and num <= 36:
            text += chr(ord('a') + num - 11)
        elif num >= 37 and num <= 62:
            text += chr(ord('A') + num - 37)
        elif num >= 63 and num <= 72:
            text += chr(ord('0') + num - 63)
        elif num >= 73 and num <= 98:
            symbols = ".,!?:;=+-*/^\\@#&()[]{}$%_`"
            text += symbols[num - 73]
        elif num == 99:
            text += ' '
    return text

def rsa_encrypt(message, N, E, block_size):
    numeric_chunks = text_to_numeric(message)
    encrypted_blocks = []
    
    for i in range(0, len(numeric_chunks), block_size):
        block = numeric_chunks[i:i+block_size]
        plaintext = int(''.join(block))
        ciphertext = modexp(plaintext, E, N)
        encrypted_blocks.append(ciphertext)
    
    return encrypted_blocks

def rsa_decrypt(encrypted_blocks, N, D, block_size):
    decrypted_numeric_chunks = []
    
    for ciphertext in encrypted_blocks:
        plaintext = modexp(ciphertext, D, N)
        plaintext_str = str(plaintext).zfill(2 * block_size)
        decrypted_numeric_chunks.extend([plaintext_str[i:i+2] for i in range(0, len(plaintext_str), 2)])
    decrypted_message = numeric_to_text(decrypted_numeric_chunks)
    return decrypted_message

#A6
def phi(n):
    p, e = factorization(n)
    result = 1

    for i in range(len(p)):
        result *= pow(p[i],e[i]) - pow(p[i], e[i]-1)

    return int(result)

def order(a, m):
    if gcd(a, m) != 1:
        return -1
    else:
        phi_m = phi(m)
        x = 1
        while True:
            if modexp(a, x, m) == 1:
                return x
            x += 1


def index(a, p, m):
    x = 0
    while True:
        if modexp(p, x, m) == a:
            return x
        x += 1

def baby_step_giant_step(a, p, m):
    n = int((m - 1)**0.5) + 1

    table1 = {i: pow(p, i, m) for i in range(n)}
    # print("Table 1:")
    # for key, value in baby_table.items():
    #    print(f"{key}: {value}")

    inv_table1 = {value: key for key, value in table1.items()}

    c = pow(pow(p, n, m), -1, m)
    table2 = {j: (a * pow(c, j, m)) % m for j in range(n)}
    # print("\nTable2 :")
    # for key, value in table2.items():
    #    print(f"{key}: {value}")

    for j, y in table2.items():
        if y in inv_table1:
            return j * n + inv_table1[y]

    return -1


def main():
    print("MAT361 Fall 2023: Program Compilation Python - Sungmin Moon")
    print("This program is uploaded in https://github.com/Elphior/MAT361_A8.git\n")
    print("Program #1: euclid(a, b)")
    print("Program #2: sieve(a, b)")
    print("Program #3: trialdivision(n)")
    print("Program #4: lineareqsolver(e)")
    print("Program #5: multinv(a, m)")
    print("Program #6: modexp(a, x, m)")
    print("Program #7: Chinese Remainder Theorem")
    print("Program #8: Caesar Shift Cipher")
    print("Program #9: Affine Cipher")
    print("Program #10: Euler's Totient (phi) Function")
    print("Program #11: RSA Encryption/Decryption")
    print("Program #12: order")
    print("Program #13: index( p^x ≡ a ( mod m ))")
    print("Program #14: Baby-step Giant-Step index( p^x ≡ a ( mod m ))")
    select = int(input("Select which program to run: "))

    if select == 1:
        print("\nSelected Euclid")
        a = int(input("Enter a positive integer a: "))
        b = int(input("Enter a positive integer b: "))

        gcd_val, x, y = euclid(a, b)
        print(f"gcd({a}, {b}): {gcd_val}")
        print(f"{x} * {a} + {y} * {b} = {x * a + y * b}")
    elif select == 2:
        print("\nSelected Sieve of Eratosthenes")
        a = int(input("Enter a positive integer a: "))
        b = int(input("Enter a positive integer b: "))

        primes = sieve(a, b)
        print(*primes)
    elif select == 3:
        print("\nSelected Trial Division")
        n = int(input("Enter a positive integer n: "))

        p, e = factorization(n)
        print("p:", *p)
        print("e:", *e)
    elif select == 4:
        print("\nSelected Linear Equation Solver")
        print("ax + by = c")
        a = int(input("Enter value for a: "))
        b = int(input("Enter value for b: "))
        c = int(input("Enter value for c: "))

        lineareqsolver(a, b, c)
    elif select == 5:
        print("\nSelected multinv")
        a = int(input("Enter value for a: "))
        m = int(input("Enter value for m: "))

        inv = multinv(a, m)
        if inv != -1:
            print(f"multinv({a}, {m}) = {inv}")
    elif select == 6:
        print("\nSelected modexp")
        a = int(input("Enter value for a: "))
        x = int(input("Enter value for x: "))
        m = int(input("Enter value for m: "))

        result = modexp(a, x, m)
        print(f"{a}^{x} mod {m} = {result}")

    elif select == 7:
        remainders = list(map(int, input("Enter a list of remainders (space-separated): ").split()))
        moduli = list(map(int, input("Enter a list of moduli (space-separated): ").split()))
        result = chinrem(moduli, remainders)
        print(f"Solution to Chinese Remainder Theorem: {result}")

    elif select == 8:
        text = input("Enter text for Caesar Shift Cipher: ")
        shift = int(input("Enter the shift value: "))
        encrypted_text = caesar_cipher(text, shift)
        print(f"Encrypted text: {encrypted_text}")
        decrypted_text = caesar_decipher(encrypted_text, shift)
        print(f"Decrypted text: {decrypted_text}")

    elif select == 9:
        task = input("Choose a task (encryption or decryption): ").strip().lower()
        if task == "encryption":
            text = input("Enter text for Affine Cipher: ")
            a = int(input("Enter the 'a' value: "))
            b = int(input("Enter the 'b' value: "))
            encrypted_text = affine_cipher(text, a, b)
            print(f"Encrypted text: {encrypted_text}")
        elif task == "decryption":
            text = input("Enter text for Affine Cipher: ")
            a = int(input("Enter the 'a' value: "))
            b = int(input("Enter the 'b' value: "))
            decrypted_text = affine_decipher(text, a, b)
            print(f"Decrypted text: {decrypted_text}")
        else:
            print("Invalid task. Please choose 'encryption' or 'decryption'.")
    elif select == 10:
        n = int(input("Enter a number to compute Euler's totient (phi): "))
        result = phi(n)
        print(f"Euler's totient (phi) of {n}: {result}")
    elif select == 11:
        task = input("Choose a task (encryption or decryption): ").strip().lower()
        if task == "encryption":
            p = int(input("Enter the value of p: "))
            q = int(input("Enter the value of q: "))
            N = p * q
            E = int(input("Enter the public encryption exponent (e): "))
            block_size = int(input("Enter the number of blocks: "))
            message = input("Enter the message to encrypt: ")
            encrypted_blocks = rsa_encrypt(message, N, E, block_size)
            print("\nEncrypted:", ' '.join(map(str, encrypted_blocks)))
            
    
        elif task == "decryption":
            p = int(input("Enter the value of p: "))
            q = int(input("Enter the value of q: "))
            N = p * q
            E = int(input("Enter the public encryption exponent (e): "))
            D = multinv(E, (p - 1) * (q - 1))
            encrypted_blocks_input = input("Enter the encrypted code (space-separated blocks): ").strip()
            encrypted_blocks = list(map(int, encrypted_blocks_input.split()))

            decrypted_message = rsa_decrypt(encrypted_blocks, N, D, len(str(encrypted_blocks[0])) // 2)
            print("\nDecrypted:", decrypted_message)
    
        else:
            print("Invalid task. Please choose 'encryption' or 'decryption'.")
    elif select == 12:
        a = int(input("Enter a: "))
        m = int(input("Enter m: "))
        result = order(a, m)
        print(f"Order of {a} (mod {m}): {result}")

    elif select == 13:
        # (3) Brute force attack: Solve p^x ≡ a (mod m)
        a = int(input("Enter a: "))
        p = int(input("Enter p: "))
        m = int(input("Enter m: "))
        start = time.time()
        result = index(a, p, m)
        end = time.time()
        if result != -1:
            runtime = end - start
            print(f"The solution to {p}^x ≡ {a} (mod {m}) is x = {result}")
            # print(f"Runtime: {runtime}")
        else:
            print(f"No solution found for {p}^x ≡ {a} (mod {m}).")
    elif select == 14:
        a = int(input("Enter a: "))
        p = int(input("Enter p: "))
        m = int(input("Enter m: "))
        start = time.time()

        result = baby_step_giant_step(a, p, m)
        end = time.time()
        if result != -1:
            runtime = end - start
            print(f"The solution to {p}^x ≡ {a} (mod {m}) is x = {result}")
            # print(f"Runtime: {runtime}")
        else:
            print(f"No solution found for {p}^x ≡ {a} (mod {m}).")
    else:
        print("\nInvalid Input!")
        print("Terminating...")

if __name__ == "__main__":
    main()
