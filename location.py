import numpy as np
def path_loss(T0,d,d0,a):
     # 计算路径损耗
    # T0: 参考路径损耗
    # d: 实际距离
    # d0: 参考距离
    # a: 路径损耗指数
   return T0*(d/d0)**(-a)
def location(K):
    #固定AP和RIS位置
    location_AP=np.array([-50,0,12])
    location_RIS=np.array([0,0,12])
    #路径损耗参数
    T0=1e-3
    a_UA = 3.6 #USER-AP
    a_UR = 2.8#USER-RIS
    a_RA = 2.2#RIS-AP
    #路径损耗计算
    dis_RA=np.linalg.norm(location_AP-location_RIS)
    pl_RA=path_loss(T0,dis_RA,1,a_RA)
    #初始化路径损耗和用户位置
    pl_UA=np.zeros(K)
    PL_UR=np.zeros(K)
    location_user=np.zeros((K,3))
    #随机生成用户位置
    for k in range(K):
        x = np.random.uniform(0, 20)
        y = (np.random.uniform(-10, 10))
        z = 0
        location_user[k, :] = [x, y, z]

    #计算用户到AP和用户到RIS的路径损害
    for k in range(K):
        dis_UA=np.linalg.norm(location_AP-location_user[k,:])
        pl_UA[k] = path_loss(T0,dis_UA,1,a_UA)

        dis_UR=np.linalg.norm(location_RIS-location_user[k,:])
        PL_UR[k] = path_loss(T0,dis_UR,1,a_UR)

    return pl_RA,pl_UA,dis_RA,dis_UA,location_user