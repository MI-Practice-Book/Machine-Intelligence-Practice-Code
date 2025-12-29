"""
数据集准备主脚本
整合下载、解压、格式转换和分析等全流程

Author: Machine Intelligence Practice
Date: 2025-01
"""

import logging
import argparse
import sys
from pathlib import Path
import tarfile
import urllib.request
from tqdm import tqdm

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger, TaskTimer
from utils.voc_converter import VOCConverter
from utils.dataset_analyzer import DatasetAnalyzer

# VOC数据集类别
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


class DownloadProgressBar(tqdm):
    """下载进度条"""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        更新进度条

        Args:
            b: 已下载的块数
            bsize: 块大小
            tsize: 总大小
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path) -> bool:
    """
    下载文件

    Args:
        url: 下载链接
        output_path: 保存路径

    Returns:
        是否下载成功
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"开始下载: {url}")
        logging.info(f"保存到: {output_path}")

        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

        logging.info(f"下载完成: {output_path}")
        return True

    except Exception as e:
        logging.error(f"下载失败: {e}")
        return False


def extract_tar_file(tar_path: Path, extract_to: Path) -> bool:
    """
    解压tar文件

    Args:
        tar_path: tar文件路径
        extract_to: 解压目标目录

    Returns:
        是否解压成功
    """
    with TaskTimer(f"解压 {tar_path.name}"):
        try:
            extract_to.mkdir(parents=True, exist_ok=True)

            logging.info(f"开始解压: {tar_path}")
            logging.info(f"解压到: {extract_to}")

            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=extract_to)

            logging.info(f"解压完成: {extract_to}")
            return True

        except Exception as e:
            logging.error(f"解压失败: {e}")
            return False


def prepare_voc_dataset(
        data_root: Path,
        year: str = '2012',
        download: bool = True,
        convert: bool = True,
        analyze: bool = True
) -> bool:
    """
    准备VOC数据集

    Args:
        data_root: 数据根目录
        year: VOC年份 (2007/2012)
        download: 是否下载数据
        convert: 是否转换格式
        analyze: 是否分析数据

    Returns:
        准备是否成功
    """
    with TaskTimer("数据集准备"):

        # 数据集URL
        if year == '2007':
            trainval_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
            test_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
        elif year == '2012':
            trainval_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
            test_url = None  # 2012没有公开测试集
        else:
            logging.error(f"不支持的年份: {year}")
            return False

        # 路径设置
        download_dir = data_root / "downloads"
        voc_dir = data_root / f"VOC{year}"

        # ========== 1. 下载数据 ==========
        if download:
            logging.info("\n" + "=" * 70)
            logging.info("阶段 1/4: 下载数据集")
            logging.info("=" * 70)

            trainval_tar = download_dir / f"VOCtrainval_{year}.tar"
            if not trainval_tar.exists():
                if not download_file(trainval_url, trainval_tar):
                    return False
            else:
                logging.info(f"文件已存在,跳过下载: {trainval_tar}")

            if test_url:
                test_tar = download_dir / f"VOCtest_{year}.tar"
                if not test_tar.exists():
                    if not download_file(test_url, test_tar):
                        return False
                else:
                    logging.info(f"文件已存在,跳过下载: {test_tar}")

            # 解压数据
            if not (voc_dir / "JPEGImages").exists():
                if not extract_tar_file(trainval_tar, data_root):
                    return False
                if test_url and test_tar.exists():
                    if not extract_tar_file(test_tar, data_root):
                        return False
            else:
                logging.info(f"数据已解压,跳过解压步骤")

        # ========== 2. 格式转换 ==========
        if convert:
            logging.info("\n" + "=" * 70)
            logging.info("阶段 2/4: VOC格式转换为YOLO格式")
            logging.info("=" * 70)

            # 创建转换器
            converter = VOCConverter(class_names=VOC_CLASSES)

            # 读取数据集划分文件
            imagesets_dir = voc_dir / "ImageSets" / "Main"

            for subset in ['train', 'val']:
                split_file = imagesets_dir / f"{subset}.txt"

                if not split_file.exists():
                    logging.warning(f"划分文件不存在: {split_file}")
                    continue

                # 读取文件列表
                with open(split_file, 'r') as f:
                    image_ids = [line.strip() for line in f.readlines()]

                logging.info(f"\n处理 {subset} 集: {len(image_ids)} 张图像")

                # 创建临时标注目录
                temp_annotations_dir = data_root / f"temp_annotations_{subset}"
                temp_annotations_dir.mkdir(parents=True, exist_ok=True)

                # 复制对应的XML文件
                annotations_dir = voc_dir / "Annotations"
                for img_id in image_ids:
                    src_xml = annotations_dir / f"{img_id}.xml"
                    dst_xml = temp_annotations_dir / f"{img_id}.xml"
                    if src_xml.exists():
                        import shutil
                        shutil.copy(src_xml, dst_xml)

                # 转换
                output_dir = data_root / "labels" / subset
                converter.convert_dataset(
                    annotations_dir=temp_annotations_dir,
                    output_dir=output_dir,
                    subset=subset
                )

                # 清理临时目录
                import shutil
                shutil.rmtree(temp_annotations_dir)

        # ========== 3. 数据分析 ==========
        if analyze:
            logging.info("\n" + "=" * 70)
            logging.info("阶段 3/4: 数据集分析")
            logging.info("=" * 70)

            analyzer = DatasetAnalyzer(class_names=VOC_CLASSES)

            for subset in ['train', 'val']:
                labels_dir = data_root / "labels" / subset

                if not labels_dir.exists():
                    logging.warning(f"标注目录不存在: {labels_dir}")
                    continue

                # 分析数据集
                stats = analyzer.analyze_dataset(
                    labels_dir=labels_dir,
                    image_size=(640, 640),
                    subset=subset
                )

                # 保存报告
                report_dir = data_root / "reports"
                report_dir.mkdir(parents=True, exist_ok=True)
                analyzer.save_report(report_dir / f"{subset}_analysis.json")

        # ========== 4. 生成配置文件 ==========
        logging.info("\n" + "=" * 70)
        logging.info("阶段 4/4: 生成YOLO配置文件")
        logging.info("=" * 70)

        yaml_content = f"""# VOC{year} Dataset Configuration for YOLOv5
# 数据集路径
path: {data_root.absolute()}  # 数据集根目录
train: labels/train  # 训练集标注路径(相对于path)
val: labels/val      # 验证集标注路径(相对于path)

# 类别数量
nc: {len(VOC_CLASSES)}

# 类别名称
names: {VOC_CLASSES}
"""

        yaml_path = data_root / f"VOC{year}.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

        logging.info(f"配置文件已生成: {yaml_path}")

        logging.info("\n" + "=" * 70)
        logging.info("✅ 数据集准备完成!")
        logging.info("=" * 70)
        logging.info(f"数据集目录: {data_root}")
        logging.info(f"配置文件: {yaml_path}")
        logging.info(f"训练集标注: {data_root / 'labels' / 'train'}")
        logging.info(f"验证集标注: {data_root / 'labels' / 'val'}")
        logging.info("=" * 70 + "\n")

        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VOC数据集准备脚本")

    parser.add_argument(
        '--data_root',
        type=str,
        default='data',
        help='数据根目录 (默认: data)'
    )

    parser.add_argument(
        '--year',
        type=str,
        default='2012',
        choices=['2007', '2012'],
        help='VOC数据集年份 (默认: 2012)'
    )

    parser.add_argument(
        '--skip_download',
        action='store_true',
        help='跳过下载步骤'
    )

    parser.add_argument(
        '--skip_convert',
        action='store_true',
        help='跳过格式转换步骤'
    )

    parser.add_argument(
        '--skip_analyze',
        action='store_true',
        help='跳过数据分析步骤'
    )

    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='日志文件路径'
    )

    args = parser.parse_args()

    # 设置日志
    log_file = args.log_file
    if log_file is None:
        log_dir = Path(args.data_root) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "prepare_dataset.log"

    setup_logger(log_file=str(log_file))

    logging.info("=" * 70)
    logging.info("VOC数据集准备工具")
    logging.info("=" * 70)
    logging.info(f"数据根目录: {args.data_root}")
    logging.info(f"数据集年份: VOC{args.year}")
    logging.info(f"日志文件: {log_file}")
    logging.info("=" * 70 + "\n")

    # 准备数据集
    data_root = Path(args.data_root)
    success = prepare_voc_dataset(
        data_root=data_root,
        year=args.year,
        download=not args.skip_download,
        convert=not args.skip_convert,
        analyze=not args.skip_analyze
    )

    if success:
        logging.info("✅ 数据集准备成功!")
        return 0
    else:
        logging.error("❌ 数据集准备失败!")
        return 1


if __name__ == "__main__":
    sys.exit(main())