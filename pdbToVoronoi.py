__author__ = "XuanYao"
__email__ = "xy2018@hnu.com"

'''
这是一个可以将蛋白质PDB格式文件转换为Voronoi图的模块

输入
    必选参数:pdb文件路径
    可选参数：可选残基文件(residue_list.csv)路径,以及可选残基中的pdb id
备注：
    可选残基文件(residue_list.csv)格式为csv文件,至少包含两个属性:pdbid,residueid
输出：
    figure1表示直接投影到平面上,该平面为使用最小二乘法拟合的平面;
    figure2表示投影到米勒圆柱面上;
    figure3将两张图共同显示在一个窗口中;
    生成的Voronoi图红色表示酸性氨基酸,蓝色表示碱性氨基酸,绿色表示其他氨基酸

使用方法为:
导入该模块
    import pdbToVoronoi
调用函数
    pdbToVoronoi.pdbToVoronoi("xxx.pdb","residue_list.csv"=None,"pdbid"=None)
生成图像
    figure1
    figure2
    figure3
'''

import numpy as np # 导入numpy包
import sys
import getopt
import math
import scipy
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,voronoi_plot_2d
from seaborn import xkcd_rgb

class coor: # 创建coor类，该类中存储三维坐标，并可根据自定义函数求其余坐标与该坐标的距离
    def __init__(self,x,y,z): # 初始化参数，将输入的三个值分别定义为self.x,self.y,self.z
        self.x=x 
        self.y=y
        self.z=z
    def distance(self,a,b,c): # 定义类中的函数distance，此函数需要输入三个参数
        dis_x=a-self.x # 三个方向上的距离分别为distance函数中的三个参数与atom类中的三个参数的差
        dis_y=b-self.y
        dis_z=c-self.z
        dis=round(np.sqrt(dis_x**2+dis_y**2+dis_z**2),3) # 距离三个方向上的差的平方根，并保留三位小数
        return dis # 返回求出的距离


class atom(coor): # 创建atom类，该类继承了coor类中的distance函数
    def __init__(self,a,f,aa,aan,x,y,z,e): # 初始化参数
        self.x=x # x为横坐标
        self.y=y # y为纵坐标
        self.z=z # z为竖坐标
        self.a=a # a为原子名称，如CA,CB,N,S
        self.f=f # f为转变位置指示，当该原子只有一个坐标时f为空，当有两个坐标时f分别为'A'和'B'
        self.aa=aa # aa为残基名称，如GLU,CYS
        self.aan=aan # aan为残基序列数目，是该原子所属的氨基酸残基在氨基酸序列中的位置
        self.e=e # e为元素标识，如C,N,O,S

def pdbToAtomList(pdbFileRead):
    atomList = []
    with open(pdbFileRead,"r") as fr:
        for line in fr:
            line = line.strip().split()
            if line[0] == 'ATOM' and line[2] == 'CA':
                a=line[2] # 该行第13:16个字符串为原子名称
                f=line[4] # 该行第17个字符为转变位置指示
                aa=line[3] # 该行第18:20个字符为残基名称
                aan=int(line[5]) # 该行第23:26个数字为残基序列数目
                x=float(line[6]) # 该行第31:38个字符为x坐标
                y=float(line[7]) # 该行第39:46个字符为y坐标
                z=float(line[8]) # 该行第47:54个字符为z坐标
                e=line[-1] # 该行第78个字符为元素标识
                atomList += [atom(a,f,aa,aan,x,y,z,e)] # 将a,f,aa,aan,x,y,z,e存入atom类中，并添加至atom_list列表中
    return atomList
def selectAtomFromResidueList(atomList,residueListFile,pdbid):
    indexList = []
    with open (residueListFile,'r') as fr:
        for line in fr:
            line = line.strip().split(',')
            if line[0] == pdbid:
                indexList.append(line[1])
    selectAtomList = []
    for i in indexList:
        selectAtomList.append(atomList[int(i)])
    return selectAtomList

def miller(x, y, z): # 该函数为米勒圆柱投影，通过将三维坐标投影在圆柱面上，并将圆柱面展开，生成二维投影坐标
    radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    latitude = math.asin(z/radius)
    longitude = math.atan2(y, x)
    lat = 5/4 * np.log(np.tan(np.pi/4 + 2/5 * latitude))
    return lat,longitude


def rod_rotate(vector,axis,angle): # 该函数为罗德里格旋转公式，将一个三维向量vector绕单位向量旋转轴axis转动angle，并生成旋转后的vector_rot
    vector_rot = math.cos(angle)*vector + (1-math.cos(angle))*(np.dot(vector,axis))*axis + math.sin(angle)*np.cross(axis,vector)
    return vector_rot


def leastSqurePlaneFitting(atomArray):
    A = np.ones((len(atomArray),3))
    b = np.zeros((len(atomArray),1))
    i = 0
    for line in atomArray:
        A[i][0],A[i][1] = line[0],line[1]
        b[i][0] = line[2]
        i+=1
    #print(len(A))
    #print(len(b))
    #通过X=(AT*A)-1*AT*b直接求解
    A_T = A.T
    A1 = np.dot(A_T,A)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2,A_T)
    X= np.dot(A3, b)
    #print('平面拟合结果为：z = %.3f * x + %.3f * y + %.3f'%(X[0,0],X[1,0],X[2,0]))
    normalVector = [X[0,0],X[1,0],-1]
    return normalVector


def coor3dTo2d(atomList,projectOption='normal'):
    # 该函数将存有附近原子类的列表转化为三维坐标矩阵与原子类型列表，并将三维坐标投影为二维坐标
    # 该函数提供了两种投影，projectOption为normal时，转动所有原子坐标，使离中心O最远的两个原子a1,a2与中心构成的平面a1Oa2与xOy平面平行，并将三维坐标
    # 投影至二维;projectOption为miller时，转动原子坐标使离中心最远的两个原子构成的向量与z轴平行，通过米勒圆柱投影将三维坐标投影至二维

    atomNum = len(atomList) # 计算原子的数目
    atomArray = np.zeros((atomNum,3)) # 生成储存原子三维坐标的零矩阵
    residueTypeList = [] # 生成储存氨基酸类型的空列表

    for i in range(atomNum):
        na = atomList[i]                    # na为atomList列表中的每一个附近原子类
        atomArray[i,:] = [na.x,na.y,na.z]   # 将na中的三维坐标存入三维坐标矩阵中
        residueTypeList += [na.aa]         # 将原子类型存入原子类型列表中
    #print(atomArray)
    #print(len(atomArray))
    # 计算所有原子的中心的坐标
    x_center = np.mean(atomArray[:,0])
    y_center = np.mean(atomArray[:,1])
    z_center = np.mean(atomArray[:,2])

    atomArray -= [x_center,y_center,z_center] # 将原子坐标中心化
    square_array = atomArray**2 # 将array中的每个值求平方
    sqrt_array = np.sqrt(sum(square_array.T)) # 将平方后的array转置求和并开方，以求每个原子离原点的距离
    index = np.argsort(sqrt_array)[::-1] # 建立索引，从左到右依次为距离最远至最近的原子在atomArray中的位置
    dis_1st = atomArray[index[0],:] # 将离原点最远的两个原子的坐标记录下来
    dis_2nd = atomArray[index[1],:]

    # 建立normal向量，通过转动该向量最终与z轴平行
    if projectOption == 'normal':
        normalVector = leastSqurePlaneFitting(atomArray) # normal向量为对两个距离最远的原子的构成的向量叉乘，生成垂直于这两个向量的法向量
    elif projectOption == 'miller':
        normalVector = dis_1st - dis_2nd # 如果用米勒圆柱投影，则normal向量为两个距离最远的原子构成的向量
    else:
        raise ValueError('No such projection option')

    # 计算转轴与转动角度
    normalVector /= np.linalg.norm(normalVector) # 将normal向量转为单位向量
    z_vector = np.array([0,0,1]) # 建立z轴上的单位向量
    angle_cos = sum(z_vector*normalVector)/np.linalg.norm(normalVector)# 求z轴方向上的单位向量与normal向量的余弦值
    angle = math.acos(angle_cos) # 根据余弦值求弧度值
    rotate_axis = np.cross(z_vector,normalVector) # 根据z轴单位向量与normal向量叉乘，生成法向量，该法向量为转轴
    rotate_axis /= np.linalg.norm(rotate_axis) # 将转轴转为单位向量

    # 通过罗德里格旋转公式计算旋转后的原子坐标
    vector_rot = np.zeros((atomNum,3)) # 生成储存旋转后的向量的矩阵
    for i in range(atomNum):
        vector = atomArray[i,:]
        vector_rot[i,:] = rod_rotate(vector,rotate_axis,angle) # 调用rod_rotate函数将atomArray中的原子坐标旋转

    # 如果投影选项为normal，则直接将三维坐标投影在xOy平面上;如果投影选项为miller，则通过米勒圆柱投影将坐标投影
    if projectOption == 'normal':
        projection = vector_rot[:,0:2]
    elif projectOption == 'miller':
        projection = np.zeros((len(vector_rot),2))
        for i in range(len(vector_rot)):
            x,y = miller(vector_rot[i,0],vector_rot[i,1],vector_rot[i,2])
            projection[i,:] = np.array((x,y))
    else:
        raise ValueError('No such projection option')
    #print(projection,residueTypeList)
    return projection,residueTypeList


def voronoi_finite_polygons_2d(vor, radius=None): 
    # vor类中的多边形顶点不包括图像外的，该函数通过输入的vor类，可输出所有的多边形顶点，用于填充颜色
    # 该函数来自https://nbviewer.jupyter.org/gist/pv/8037100
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2
    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2),(v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all([v >= 0 for v in vertices]):
            # finite region
            new_regions.append(vertices)
            continue
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def atom_color(residueTypeList,more_color='no'): 
    # 该函数通过输入原子类型列表，生成原子颜色列表，每一种颜色代表一种原子，并且同种的原子颜色也有区别

    # 创建四种基础颜色列表，每个列表的颜色为挑选的较易分辨的颜色
    green = ['pine green','cool green','light bluish green','kelley green','seafoam green','shamrock green','weird green','greenish',
             'jade green','algae green','minty green','seaweed green','tealish green','dark seafoam green','jungle green','dark mint green',
             'evergreen','irish green','kelly green','emerald green','bright light green','darkish green','deep green','medium green',
             'british racing green','forest green','dark forest green','darkgreen','bright green','hot green','electric green','fluro green',
             'radioactive green','true green'] # 34种绿色用于代表C原子
    blue = ['light royal blue','strong blue','primary blue','pure blue','cobalt blue','true blue','rich blue','vivid blue','vibrant blue',
            'electric blue','dodger blue','clear blue','bright blue','deep sky blue','medium blue','sky blue','cool blue','mid blue',
            'windows blue'] # 19种蓝色用于代表N原子
    red = ['dull red','deep red','red','reddish','pale red','brick red','tomato red','orange red','cherry red','lightish red','bright red',
           'darkish red','blood red','fire engine red'] # 14种红色用于代表S原子
    yellow = ['orange yellow','yellow orange','yellowish orange','golden yellow','sand yellow','sunflower yellow','sun yellow','sandy yellow',
              'yellowish','sunny yellow','butter yellow','bright yellow','canary yellow','yellowish tan','lemon yellow',
              'banana yellow'] # 16种黄色用于代表O原子

    # 如果开启more_color选项，则将seaborn中的四种颜色添加至颜色列表后
    if more_color =='yes':
        color = xkcd_rgb # 从seaborn包中导入字典xkcd_rgb
        green  += [c for c in color.keys() if 'green' in c and 'yellow' not in c ] # C
        blue   += [c for c in color.keys() if 'blue' in c and 'green' not in c ] # N
        red    += [c for c in color.keys() if 'red' in c  ] # S
        yellow += [c for c in color.keys() if 'yellow' in c and 'green' not in c] # O

    # 生成颜色类型列表，该列表中储存每个原子区域应该填充的颜色，如40个原子，有23个C，8个N，5个O，4个S，则分别从四个颜色列表中选出前23,8,5,4个颜色
    atomColorList=[] 
    j1,j2,j3,j4=0,0,0,0
    for i in residueTypeList:
        if i in ['ASP','GLU']:
            atomColorList += [red[3]] # 酸性氨基酸用红色表示
            j1 += 1
        elif i in ['ARG','HIS','LYS']:
            atomColorList += [blue[2]] # 碱性氨基酸用蓝色表示
            j2 += 1
        else:
            atomColorList += [green[0]] # 其他用绿色表示
            j3 += 1
    
    return atomColorList

def coor_to_voronoi(projection,atomColorList,plot_option='finite'): # 该函数将投影后的二维坐标绘制为vonoroi图

    vor = Voronoi(projection) # 调用scipy.spatial包中的vororoi函数，返回一个vor类

    # 画图选项为'original'时，使用scipy.spatial中的函数画图，该方法不能将所有区域填充，因为vor类中用-1代替图像外的多边形顶点
    if plot_option == 'original': 
        fig = voronoi_plot_2d(vor) # 该函数为scipy.spatial包中的函数
        i=-1
        for region in vor.regions:
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                i += 1
                return plt.fill(*zip(*polygon),color = xkcd_rgb[atomColorList[i]])

    # 画图选项为'finite'时，使用函数求出vor中图像外的多边形顶点，因此可将所有区域填充，但在某些情况下该方法会报错
    # 可能原因是投影后的二维坐标相距太近会导致报错。此方法报错时，将画图选项改为'original'可画出有缺陷的vonoroi图
    elif plot_option == 'finite':
        regions, vertices = voronoi_finite_polygons_2d(vor) # 该函数为自定义函数，源码来自https://nbviewer.jupyter.org/gist/pv/8037100
        i=0
        for region in regions:
            polygon = vertices[region]
            plt.fill(*zip(*polygon),color = xkcd_rgb[atomColorList[i]])
            i += 1
        ##plt.axis('equal') 
        plt.xlim(vor.min_bound[0] - 0.2, vor.max_bound[0] + 0.2)
        plt.ylim(vor.min_bound[1] - 0.2, vor.max_bound[1] + 0.2)
        return plt.plot(vor.points[:,0], vor.points[:,1], 'ko',markersize=3)
        
    else:
        raise ValueError('No such plot option')

def pdbToVoronoi(pdbFile,residueListFile=None,pdbid=None):
    atomList = pdbToAtomList(pdbFile)
    if residueListFile:
        if residueListFile[-3:] !='csv':
            raise Exception('residue_list文件必须是csv格式')
        elif pdbid == None:
            raise Exception('必须输入pdbid')
        else:
            selectAtomList = selectAtomFromResidueList(atomList,residueListFile,pdbid)
            normalProjection,residueTypeList = coor3dTo2d(selectAtomList,'normal') # 调用coor3dTo2d函数将atomList通过normal投影生成存储二维坐标的矩阵normalProjection
            millerProjection,residueTypeList = coor3dTo2d(selectAtomList,'miller')
            atomColorList = atom_color(residueTypeList) # 调用atom_color，输入储存原子类型的列表，生成储存每个原子对应颜色的列表
            #print(atomColorList)
            plt.figure(1),coor_to_voronoi(normalProjection,atomColorList),plt.title('normal projection') # 画出normal投影的vonoroi图

            plt.figure(2),coor_to_voronoi(millerProjection,atomColorList),plt.title('miller projection') # 画出miller投影的vonoroi图

            plt.figure(3) # 将两种投影的vonoroi图画在一张图上
            plt.subplot(1,2,1),coor_to_voronoi(normalProjection,atomColorList),plt.title('normal projection')
            plt.subplot(1,2,2),coor_to_voronoi(millerProjection,atomColorList),plt.title('miller projection')
            plt.show() # 显示图
    else:
        normalProjection,residueTypeList = coor3dTo2d(atomList,'normal') # 调用coor3dTo2d函数将atomList通过normal投影生成存储二维坐标的矩阵normalProjection
        millerProjection,residueTypeList = coor3dTo2d(atomList,'miller')
        atomColorList = atom_color(residueTypeList) # 调用atom_color，输入储存原子类型的列表，生成储存每个原子对应颜色的列表
        #print(atomColorList)
        plt.figure(1),coor_to_voronoi(normalProjection,atomColorList),plt.title('normal projection') # 画出normal投影的vonoroi图

        plt.figure(2),coor_to_voronoi(millerProjection,atomColorList),plt.title('miller projection') # 画出miller投影的vonoroi图

        plt.figure(3) # 将两种投影的vonoroi图画在一张图上
        plt.subplot(1,2,1),coor_to_voronoi(normalProjection,atomColorList),plt.title('normal projection')
        plt.subplot(1,2,2),coor_to_voronoi(millerProjection,atomColorList),plt.title('miller projection')
        plt.show() # 显示图



if __name__=='__main__':
    #pdbToVoronoi("F:/test/1a2y.pdb")
    #pdbToVoronoi("F:/test/1a2y.pdb",'F:/test/residue_list.csv','1a2y')
    #pdbToVoronoi("F:/test/1a2y.pdb",'F:/test/residue_list.pdb','1a2y')
    pdbToVoronoi("F:/test/1a2y.pdb",'F:/test/residue_list.csv')

