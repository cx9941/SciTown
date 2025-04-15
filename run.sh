model_name='qwen'

cd src

# for query in '城市治理' '极端气候应对' '智慧医疗系统' '灾害预警与响应系统' '水生生态系统治理' '深海探测' '小行星采样与资源开发' '脑-机接口系统研究'
for query in  '记忆重建与虚拟现实融合研究'
do
python -m test_researcher.main --query $query --model_name $model_name > ../logs/$model_name/$query.log &
done