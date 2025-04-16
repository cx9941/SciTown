model_name='qwen'

cd src

for query in '城市治理' '极端气候应对' '智慧医疗系统' '灾害预警与响应系统' '水生生态系统治理' '深海探测' '小行星采样与资源开发' '脑-机接口系统研究' '记忆重建与虚拟现实融合研究' '可再生能源智能调度系统' '假设一年后小行星撞击地球，如何解决？' '深海无人智能协同作业系统' '轨道空间碎片智能识别与自清除系统' '跨物种语言理解与交流系统构建'
# for query in '记忆重建与虚拟现实融合研究'
do
python -m test_researcher.main --query $query --model_name $model_name > ../logs/$model_name/$query.log &
done