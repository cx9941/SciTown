model_name='deepseek-v3'

cd src

for query in '深海探测'
do
python -m test_researcher.main --query $query --model_name $model_name > ../logs/$model_name/$query.log &
done