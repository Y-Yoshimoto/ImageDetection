#/bin/bash
## 画像のファイル名を連番にする。
# 作業対象ディレクトリを指定
org_dir="./images"
export_dir="./images_rename"
# 出力先ディレクトリを作成
mkdir -p $export_dir

# 画像ファイルを連番にリネーム
i=0
for file in ${org_dir}/*.jpg; do
    # 拡張子を取得
    ext=${file##*.}
    # ファイル名を取得
    filename=`basename ${file} .${ext}`
    # ファイル名を連番にリネーム
    new_filename=$(printf "%04d.${ext}" ${i})
    cp ${file} ${export_dir}/${new_filename}
    i=`expr $i + 1`
done
