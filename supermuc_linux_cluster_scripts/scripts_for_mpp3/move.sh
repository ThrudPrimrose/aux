#convert from string to array
IFS=' ' read -r -a jobtypes <<< "$s_jobtypes"

if [ -d ${jobscriptdir} ]
then
    echo "${jobscriptdir} already exists"
else
    echo "no sense if ${jobscriptdir} are not generated"
    exit
fi

cd ${jobscriptdir}

#copy executables to alreay created files by the generator.sh
for t in ${jobtypes[@]}
do
    for i in pond-${t}-*/
    do
        echo $i
        cp ../pond-${t} "$i"
    done
done

cd ..
