#!/bin/bash
input="$1"
output="$2"
codedDim=32

ffmpeg -i "$input" -vf "pad=if(mod(iw\,${codedDim})\,ceil(iw/${codedDim})*${codedDim}\,iw):if(mod(ih\,${codedDim})\,ceil(ih/${codedDim})*${codedDim}\,ih)" -c:v libx265 -preset slow -an "$output"
