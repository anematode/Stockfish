# Full disclosure: Vibed by Gemini
# USAGE: python3 gen.py > goober.inc

def generate_cxx_code(M, N):
    """
    Generates the C++ code snippet for a given M (adds) and N (subs).
    """
    # Template arguments: M 'Add's followed by N 'Sub's
    template_args = ', '.join(['Add'] * M + ['Sub'] * N)

    # Function call arguments: M 'added[i]' followed by N 'removed[j]'
    added_args = [f"added[{i}]" for i in range(M)]
    removed_args = [f"removed[{j}]" for j in range(N)]
    function_args = ', '.join(added_args + removed_args)

    # The full C++ block
    code = (
        f"if constexpr (M == {M} && N == {N}) {{\n"
        f"    ctx->template apply_threats<{template_args}>({function_args});\n"
        f"}}"
    )
    return code

n = 15

def generate_all_blocks():
    """
    Generates and prints all C++ code blocks for 1 <= M+N <= n.
    """
    # A list to store all generated code blocks
    all_code_blocks = []

    # Iterate over all possible sums 1 to n
    for total in range(0, n + 1):
        # Iterate over M from 0 up to the total
        for M in range(total + 1):
            N = total - M
            # Ensure M >= 0 and N >= 0 are naturally handled by the loop/math

            # The prompt example shows M=3 (adds), N=2 (subs).
            # We assume M and N can be 0, as long as M+N >= 1.

            code_block = generate_cxx_code(M, N)
            if code_block:
                all_code_blocks.append(code_block)

    # Print all generated code blocks with a separator
    print("\n\n// --- Generated C++ Code Blocks ---")
    print(' else '.join(all_code_blocks))
    print(' else { static_assert(false); }')
    print("\n// ---------------------------------")

# Run the generation function
generate_all_blocks()