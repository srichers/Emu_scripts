# make movies for each off-diagonal neutrino & anti-neutrino pair
#python3 1d_movie_offdiag.py -o 1d_offdiag_emu -p 8 -c "emu" &&
#python3 1d_movie_offdiag.py -o 1d_offdiag_etau -p 8 -c "etau" &&
#python3 1d_movie_offdiag.py -o 1d_offdiag_mutau -p 8 -c "mutau" &&
# make movie for all off-diagonal neutrino pairs
#python3 1d_movie_offdiag.py -o 1d_offdiag_nu -p 8 -c "emu" "etau" "mutau" &&
# make movie for all off-diagonal anti-neutrino pairs
#python3 1d_movie_offdiag.py -o 1d_offdiag_nubar -p 8 -c "emubar" "etaubar" "mutaubar" &&
python3 1d_movie_offdiag.py -o 1d_offdiag_uu_tt_ut -p 16 -c "mumu" "tautau" "d_mutau" "m_mutau" "mutau"