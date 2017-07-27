<template>
	<section>
		<!--工具条-->
		<el-row>
		<el-col :span="24" class="toolbar" style="padding-bottom: 0px;">
			<el-form :inline="true">
				<el-form-item>
				模型：
				</el-form-item>
				<el-form-item>
					<el-select v-model="model_value" placeholder="选择预测模型" @change="changeModel">
    					<el-option
     		 				v-for="item in model_options"
      							:key="item.model_value"
      							:label="item.model_label"
      							:value="item.model_value">
   		 				</el-option>
 					</el-select>
				</el-form-item>
				<el-form-item>
				预测集：
				</el-form-item>
				<el-form-item>
					<el-select v-model="data_value" placeholder="选择预测数据集">
    					<el-option
     		 				v-for="item in data_options"
      						:key="item.data_value"
							:label="item.data_label"
							:value="item.data_value">
						</el-option>
 					</el-select>
				</el-form-item>
				<el-form-item>
				模型参数：
				</el-form-item>
				<el-form-item>
					<el-select v-model="modeldata_value" placeholder="选择模型参数">
    					<el-option
     		 				v-for="item in modeldata_options"
      						:key="item.modeldata_value"
							:label="item.modeldata_label"
							:value="item.modeldata_value">
						</el-option>
 					</el-select>
				</el-form-item>
				<el-form-item>
					<el-button style="margin-left: 10px;"
					type="primary" :loading="ispreding" @click="predModel">预测</el-button>
				</el-form-item>
			</el-form>
		</el-col>

		<!--图-->
		<el-col :span="24" class = "chart-container">
            <div id="chartLine" style="width:100%;  display:none; height:400px;"></div>
        </el-col>
        <div id="predresult" style="display:none; width:100%; text-align:center;">
            <el-col :span="24" class="toolbar">
                <span style=" font-size:16px;">{{ predresulttext }}</span>
            </el-col>
        </div>
        </el-row>
	</section>
</template>

<script>
	import echarts from 'echarts'
	import { serverbase } from '../../api/api';
  export default {
    data() {
      return {
        model_options: [],
        model_value: '',
        data_options: [],
        data_value: '',
        modeldata_value: '',
        modeldata_options: [],
        csv_data: [],
        ispreding: false,	
        chartLine: null,
        predresult: null,
        predresulttext: null,
      }
    },
    methods: {
    	predModel(){
    		this.ispreding = true;
    		fetch(serverbase() + 'pred/' + '?name=' + encodeURIComponent('pred') + '&model=' + encodeURIComponent(this.model_value) + '&data=' + encodeURIComponent(this.data_value) + '&modeldata=' + encodeURIComponent(this.modeldata_value))
            .then(response => response.json())
            .then(data => {
                this.csv_data = data[0].data;
                var vLineChart=document.getElementById("chartLine");
                vLineChart.style.display = "";
                this.drawLineChart();
                var vpredresult=document.getElementById("predresult");
                this.predresulttext = "预测结果：";
                if(data[0].pred == 1){
                    this.predresulttext += "涨";
                }
                else if(data[0].pred == -1){
                    this.predresulttext += '跌';
                }
                else{
                    this.predresulttext += '持平';
                }
                vpredresult.style.display = "";
                this.ispreding = false;
            });
    	},

    	getModedataList(url){
    		fetch(serverbase() + 'pred/' + '?name=' + encodeURIComponent('modeldataset') + '&model=' + encodeURIComponent(url))
    		.then(response => response.json())
  			.then(data => {
  				this.modeldata_options = data;
    		});
    	},

    	changeModel(value){
            this.modeldata_value = '';
    		this.getModedataList(this.model_value);
    	},

    	getModelList(){
    		fetch(serverbase() + 'pred/' + '?name=' + encodeURIComponent('modelset'))
    		.then(response => response.json())
  			.then(data => {
  				this.model_options = data;
    		});
    	},

    	getTestdatasetList(){
			fetch(serverbase() + 'pred/' + '?name=' + encodeURIComponent('dataset'))
    		.then(response => response.json())
  			.then(data => {
  				this.data_options = data;
    		});
    	},

    	drawLineChart() {
            this.chartLine = echarts.init(document.getElementById('chartLine'));
            this.chartLine.setOption({
                title: {
                    left: 'center',
                    text: '股价走势'
                },
                tooltip: {
                    trigger: 'axis'
                },
                grid: {
                    left: '3%',
                    right: '4%',
                    bottom: '3%',
                    containLabel: true
                },
                xAxis: {
                    type: 'category',
                    boundaryGap: false,
                    data : this.csv_data[0].date
                },
                yAxis: {
                    type: 'value',
                    scale: true	
                },
                series: [
                    {
                        name: '股价',
                        type: 'line',
                        stack: '总量',
                        data : this.csv_data[0].price
                    },
                ]
            });   
        }
    },
    mounted: function () {
    	this.getModelList();
    	this.getTestdatasetList();
    },
    updated: function () {
    }
  }
</script>

<style scoped>
	.chart-container {
        width: 100%;
        float: left;
    }
</style>