# sheet.sh NAME: a contact sheet of 6 frames across the clip, for checking
FF="C:/Users/rushi/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.2-full_build/bin/ffmpeg.exe"
F="E:/AI/tessa/video/film/assets/footage/$1.webm"
"$FF" -v error -i "$F" -vf "fps=6/${2:-30},scale=640:-1,tile=2x3" -frames:v 1 -y "C:/Users/rushi/AppData/Local/Temp/claude/E--AI-tessa/f2242aeb-2298-4e19-9809-6668168f750e/scratchpad/sheet-$1.png"
