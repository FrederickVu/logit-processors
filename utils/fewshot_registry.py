import sys

"""
Here we hold the registry for fewshot formats.
"""

def format_hendrycks(user_input: str):
    return f"Problem:\n{user_input}\n\nSolution:"

def build_hendrycks_messages(n: int):
    samples = [
        {
            "problem": r"Find the domain of the expression  $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}",
            "solution": r"The expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
        },
        {
            "problem": r"If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$",
            "solution": r"We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}.\nFinal Answer: The final answer is 24. I hope it is correct.",
        },
        {
            "problem": r"Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": r"If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is 16. I hope it is correct.",
        },
        {
            "problem": r"If the system of equations\n\n$\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\end{align*}$ has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": r"If we multiply the first equation by $-\frac{3}{2}$, we obtain\n$$6y-9x=-\frac{3}{2}a.$$ Since we also know that $6y-9x=b$, we have\n$$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.\nFinal Answer: The final answer is $-\frac{2}{3}$. I hope it is correct.",
        }
    ]
    fewshot_turns = []
    n = n if n is not None else len(samples)
    for samp in samples[:n]:
        user_message = {
            "role": "user",
            "content": format_hendrycks(samp['problem'])
        }
        asst_message = {
            "role": "assistant", 
            "content": samp['solution']
        }
        fewshot_turns.append(user_message)
        fewshot_turns.append(asst_message)
    return fewshot_turns

FEWSHOT_REGISTRY = {
    "hendrycks": build_hendrycks_messages,

}
FEWSHOT_FORMATTING = {
    "hendrycks": format_hendrycks
}

def create_base_context(system_prompt, fewshot, fewshot_num):
    """
    Constructs a list of initial context messages (system prompt, few-shot examples).
    This will be prepended to the conversation state in ChatSession.
    
    Args:
        args: The parsed arguments object containing:
            - system_prompt (str): optional string for the system prompt
            - fewshot (str): the name of a few-shot scenario in FEWSHOT_REGISTRY
            - fewshot_num (int): optional number of fewshot examples to use
    
    Returns:
        A list of messages, each a dict like {"role": "system"/"user"/"assistant", "content": "..."}.

    Notes:
        FEWSHOT_REGISTRY (dict): an optional registry for few-shot scenarios.
            The registry is expected to map strings -> a callable that produces
            a list of messages in the format [{"role": "...", "content": "..."}].
    """
    base_context = []

    system_prompt = system_prompt
    if system_prompt and system_prompt.strip():
       base_context.append({"role": "system", "content": system_prompt.strip()})

    fewshot_name = fewshot
    fewshot_num = fewshot_num
    fewshot_messages = []
    if fewshot_name is not None:
        if fewshot_name in FEWSHOT_REGISTRY:
            fewshot_messages = FEWSHOT_REGISTRY[fewshot_name](fewshot_num)
            base_context.extend(fewshot_messages)
        else:
            print(f"Fewshot name {fewshot_name} is not in FEWSHOT_REGISTRY.")
            sys.exit(1)
    
    return base_context