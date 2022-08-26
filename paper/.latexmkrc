# Ensure './texmf//' is in '$TEXINPUTS'.
ensure_path( 'TEXINPUTS', './texmf//' );

# Use lualatex with latexmk.
$pdf_mode = 4;

# Disable postscript and dvi output.
$postscript_mode = $dvi_mode = 0;

# Add common patterns for tex engines.
set_tex_cmds( '-synctex=1 %O %S' );

# Always try to embed fonts, ignoring licensing flags, etc.
$xdvipdfmx = 'xdvipdfmx -E -o %D %O %S';

# Files to clean.
$clean_ext = 'bbl glo gls hd loa run.xml thm xdv';
