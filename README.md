# **YouTube Revenue Prediction**

1.Introduction
* 目標:建立一個可以用來預測YouTube影片收益的模型。
* 解決問題:Youtuber收入不穩定，影片上傳後要等一陣子才能知道收益如何，而官方提供的預估收益又與實際收益相差很多(如下圖)，在規劃金錢使用上會變得很麻煩，因此想用機器學習預測收入的方式解決這個問題。
<img src="https://i.imgur.com/h3VqGKK.jpg" width="300px">



2.Dataset Resource
* Dataset來自我Youtube頻道的後臺數據，即Youtube Studio，關於影片的各種訊息會被存在這個網站的資料庫中，可以匯出成csv檔，而我從中選擇了觀看次數、觀看時間、喜歡、不喜歡、分享次數、獲得的訂閱、留言數作為預測收益的feature。
![](https://i.imgur.com/GjRzrfO.png)


3.Preprocessing 
![](https://i.imgur.com/xa7wWW0.png)
* 由上圖看出留言數與收益並無直接關係，因此決定捨棄這個feature。
(其餘圖片已於Presentaion呈現，為節省版面就不附上了)
* 經測試發現分享次數對於準確率提升無幫助，因此捨棄。
* 由於各項數據的單位差距太大，因此使用preprocessing.scale進行normalize。
![](https://i.imgur.com/3HaS19l.png)

4.Models
* Decision Tree、Random Forest、KNN:將收益以1000元為區間分為四個Label進行預測。
* Linear Regression:預測收益的精準數字。

5.Result
* Decision Tree:雖然準確率不太高，但可以用來分析各feature的重要性。
![](https://i.imgur.com/FtXN8bJ.png)
![](https://i.imgur.com/yRYGIrR.png)
* Random Forest: n_estimators設為200
![](https://i.imgur.com/P1IrN8q.png)
* KNN: n_neighbors設為7
![](https://i.imgur.com/LDNrjg0.png)
* Linear Regression: 平均誤差為27%。
![](https://i.imgur.com/VYuROok.png)







6.Conclusion


Model預測的比起YT網站提供的預測來得更準確一些，因此確實可以用來幫助分析收益，但由於我自己的dataset中能預測的收益範圍有限(不夠紅QQ)，因此只適合用以預測收入在4000以下的Youtuber，若是能認識更有名的Youtuber，得到不同範圍的dataset，或許能讓應用變得更廣泛。

7.Application
* 幫朋友預測收益:從Youtuber朋友的影片數據中為其預估收益
![](https://i.imgur.com/dEgT5Ir.png)











