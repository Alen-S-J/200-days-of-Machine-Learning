def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):
    # Calculate P(A|B)
    p_a_given_b = (p_b_given_a * p_a) / ((p_b_given_a * p_a) + (p_b_given_not_a * (1 - p_a)))
    return p_a_given_b

# Example probabilities
p_a = 0.4  # P(A)
p_b_given_a = 0.7  # P(B|A)
p_b_given_not_a = 0.2  # P(B|¬A)

# Calculate P(A|B)
p_a_given_b = bayes_theorem(p_a, p_b_given_a, p_b_given_not_a)
print("P(A|B):", p_a_given_b)
