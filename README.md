# highway-env

conda env 새로 만들어주기 (highwayEnv)

pip install -r requirements.txt

#pip install stable-baselines3[extra]
or pip install stable-baselines3’[extra]’
pip install do-mpc

git clone https://github.com/eleurent/highway-env.git
git clone https://github.com/do-mpc/do-mpc.git


/home/kist-robot2/Desktop/CER_highway/highway-env/highway_env 에서
highway_env 삭제하고 인수인계파일에 있는 걸로 바꿔주기 

/home/anaconda3/envs/"highwayEnv"/lib/python3.8/site-packages/stable_baselines3 에서 
stable_baselines3 삭제 하고 인수인계 파일에 있는 stable_baselines3 로 바꿔주기 

/home/kist-robot2/Desktop/CER_highway/highway-env/script 내부에
인수인계 파일 내부에 있는 script 안의 내용물 다 넣어주기 


** 코드 실행 시 td_ver'x' 에서 decison_buffer 에 대한 ㄴ내용이 없다는 에러가 뜰 수 있음. 
이때는 decision_buffer import 해 주는 부분 모두 삭제하기 


yj_cer.py -> CER + TD3
yj_cermpc.py -> CER+TD3+MPC
yj_mpc.py -> MPC
(위의 .py 파일들은 Git 다운 후 script 폴더에 넣어서 실행. 코드 내부 경로는 각자 컴터에 맞게 변경해야함)
