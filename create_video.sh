#!/bin/sh

echo "START";

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P );
# echo $parent_path;

cd "$parent_path/images/";

for dir in *; do

	# echo $dir;

	ffmpeg_command="ffmpeg -framerate 20 -pattern_type glob -i '${dir}/*.png' -c:v libx264 -pix_fmt yuv420p ${dir}.mp4";	
	echo ${ffmpeg_command};

	eval "$ffmpeg_command";

done

echo "END";