"""
Problem No: 001
Title: Remove vowels from a string

Goal:
Practice basic Python string iteration, conditional logic, and result accumulation.

Key Concepts:
- for loop
- if condition
- string accumulation
- function return

Example:
Input: "hello world"
Output: "hll wrld"
"""


def remove_vowels(s: str) -> str:
    result = ""

    for char in s:
        if char not in "aeiou":
            result += char

    return result


if __name__ == "__main__":
    print(remove_vowels("hello world"))