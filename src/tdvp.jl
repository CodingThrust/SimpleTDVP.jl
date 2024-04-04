# density matrix
#    |    |   
# -- o -- o --
#    |    |   


# "DM-MPS" with d = 4
#    ||    ||   
# -- oo -- oo --

# Apply U = e^{-iHt} to DM-MPS (U ⊗ U')
#    ||    ||
# -- oo -- oo --
#    ||    ||
# -- oo -- oo --

# Apply Lindblad to DM-MPS
# LρL' - 0.5 * (L'Lρ - ρL'L')
# -> L ⊗ L' - 0.5 * (L'L ⊗ I - I ⊗ L'L')
#
#    ||   
#  L'L ⊗ I
#    ||              ||
# -- oo ------------ oo --