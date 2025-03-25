#
# convert match*.jpg files into a mp4 video file
#
export list="match*.jpg"
export op="ln -s"
export file=x
export ext=.jpg
export i=0

ls ${list} | while read line
do
    i=`expr "$i" + "1"`

    il="$i";

    if [ $i -lt 1000 ]; then export il="0$i"; fi
    if [ $i -lt 100 ]; then export il="00$i"; fi
    if [ $i -lt 10 ]; then export il="000$i"; fi

    ${op} $line ${file}${il}${ext}
done

outputfile=video-`date "+%y-%m-%d-%H-%M-%S"`.mp4

ffmpeg -loglevel quiet -framerate 8 -i ${file}%04d.jpg -c:v libx264 \
   -profile:v high -crf 20 -pix_fmt yuv420p $outputfile

rm ${file}*${ext}
