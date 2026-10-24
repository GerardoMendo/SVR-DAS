from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient
import os

def download_DAS_data(file,target_dir,system="Nnet"):

	if(system == "Nnet"):

		ssh = SSHClient()
		ssh.set_missing_host_key_policy(AutoAddPolicy())
		
		ssh.connect(hostname="172.17.3.65",username="gmendo91",password=os.environ.get("PASS_DASIP"))
		source_dir = '/work1/Data/OBCS-DAS/N-net/N-net_2024-08-02'

		with SCPClient(ssh.get_transport()) as scp:
			scp.get(f"{source_dir}/{file}",target_dir)

		#print(f"Downloaded file: {file}")

	elif(system == "Sanriku1"):

		ssh = SSHClient()
		ssh.set_missing_host_key_policy(AutoAddPolicy())
		ssh.connect(hostname="172.17.3.29",username="gmendo91",password=os.environ.get("PASS_SYN2"))
		source_dir = '/Data/Kamaishi/ERI_Initial_Acquistion_GL100m_2mOCP_FS800Hz/ERI_GL100_2022-02-28T08_49_04Z'

		with SCPClient(ssh.get_transport()) as scp:
			scp.get(f"{source_dir}/{file}",target_dir)

		#print(f"Downloaded file: {file}")		