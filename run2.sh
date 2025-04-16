model_name='deepseek-v3'

cd src

for query in '脑-机接口系统研究' '可再生能源智能调度系统' '假设一年后小行星撞击地球，如何解决？' '轨道空间碎片智能识别与自清除系统'
do
python -m test_researcher.main --query $query --model_name $model_name > ../logs/$model_name/$query.log &
done