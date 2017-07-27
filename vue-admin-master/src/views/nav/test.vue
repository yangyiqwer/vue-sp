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
					<el-select v-model="model_value" placeholder="选择测试模型" @change="changeModel">
						<el-option
							v-for="item in model_options"
								:key="item.model_value"
								:label="item.model_label"
								:value="item.model_value">
						</el-option>
					</el-select>
				</el-form-item>
				<el-form-item>
				测试集：
				</el-form-item>
				<el-form-item>
					<el-select v-model="data_value" placeholder="选择测试数据集">
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
					type="primary" :loading="istesting" @click="testModel">测试</el-button>
				</el-form-item>
			</el-form>
		</el-col>

		<!--图-->
		<el-col :span="24" class = "chart-container">
			<div id="chartLine" style="width:100%; height:400px;"></div>
		</el-col>
		<el-col :span="24" class = "chart-container">
			<div id="chartRealHistogram" style="width:100%; display:none; height:200px;"></div>
		</el-col>
		<el-col :span="24" class = "chart-container">
			<div id="chartPredHistogram" style="width:100%; display:none; height:200px;"></div>
		</el-col>
		<el-col :span="24" class = "chart-container">
			<div id="chartWrongHistogram" style="width:100%; display:none; height:120px;"></div>
		</el-col>
		<div id="testresult" style="display:none; width:100%; text-align:center;">
			<el-col :span="24" class="toolbar">
				<span style=" font-size:16px;">{{ testresulttext }}</span>
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
		istesting: false,	
		chartLine: null,
		chartRealHistogram: null,
		chartPredHistogram: null,
		chartWrongHistogram: null,
		test_data: [],
		test_text: [],
		testresult: null,
		testresulttext: null,
	  }
	},
	methods: {
		testModel(){
			this.istesting = true;
			fetch(serverbase() + 'test/' + '?name=' + encodeURIComponent('test') + '&model=' + encodeURIComponent(this.model_value) + '&data=' + encodeURIComponent(this.data_value) + '&modeldata=' + encodeURIComponent(this.modeldata_value))
			.then(response => response.json())
			.then(data => {
				this.csv_data = data[0].data;
				this.test_data = data[0].Trend;
				this.test_text = data[0].text;
				var vLineChart=document.getElementById("chartLine");
				vLineChart.style.display = "";
				this.drawLineChart();
				var vchartRealHistogram=document.getElementById("chartRealHistogram");
				chartRealHistogram.style.display = "";
				this.drawchartRealHistogram();
				var vchartPredHistogram=document.getElementById("chartPredHistogram");
				chartPredHistogram.style.display = "";
				this.drawchartPredHistogram();
				var vchartWrongHistogram=document.getElementById("chartWrongHistogram");
				chartWrongHistogram.style.display = "";
				this.drawchartWrongHistogram();
				var vtestresult=document.getElementById("testresult");
				this.testresulttext = "总数: " + this.test_text[0].total + " 正确率: " + this.test_text[0].Correct_rate + '%';
				vtestresult.style.display = "";
				this.istesting = false;
			});
		},

		getModedataList(url){
			fetch(serverbase() + 'test/' + '?name=' + encodeURIComponent('modeldataset') + '&model=' + encodeURIComponent(url))
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
    		fetch(serverbase() + 'test/' + '?name=' + encodeURIComponent('modelset'))
    		.then(response => response.json())
  			.then(data => {
  				this.model_options = data;
    		});
    	},

    	getTestdatasetList(){
			fetch(serverbase() + 'test/' + '?name=' + encodeURIComponent('dataset'))
    		.then(response => response.json())
  			.then(data => {
  				this.data_options = data;
    		});
    	},

    	drawchartWrongHistogram(){
			this.chartWrongHistogram = echarts.init(document.getElementById('chartWrongHistogram'));
			this.chartWrongHistogram.setOption({
				title: {
	                	left: 'center',
	                    text: '预测正确分析'
	                },
				    tooltip : {
				        trigger: 'axis',
				        axisPointer : {
				            type : 'shadow'
				        },
				        formatter: function (params) {
			           		var res = params[0].axisValue + "<br/>" +params[0].marker + params[0].seriesName + ' : ';
			           		if(params[0].value == 1){
			           			res += '错误';
			           		}
			           		else if(params[0].value == 0){
			           			res += '正确';
			           		}
			           		return res;
			        }
				    },
				    grid: {
				    	top: '25%',
				        left: '3%',
				        right: '4%',
				        bottom: '8%',
				        containLabel: true
				    },
				    xAxis : [
				        {
				            type : 'category',
				            data : this.test_data.date,
				            axisTick: {
				                alignWithLabel: true
				            }
				        }
				    ],
				    yAxis : [
				        {
				            type : 'value',
				            interval : 1,
				            axisLabel: {
				                formatter: function (value, index) {
				                    if(value == 1) return '错';
				                    else if(value == 0) return'对';
				                }
				            }
				        }
				    ],
				    series : [
				        {
				            name:'预测结果',
				            type:'bar',
				            barWidth: '60%',
				            data:this.test_data.wrongpred
				        }
				    ]
			});
		},

		drawchartPredHistogram(){
			this.chartPredHistogram = echarts.init(document.getElementById('chartPredHistogram'));
			this.chartPredHistogram.setOption({
					color: ['#3398DB'],
					title: {
	                	left: 'center',
	                    text: '预测涨跌走势'
	                },
				    tooltip : {
				        trigger: 'axis',
				        axisPointer : {
				            type : 'shadow'
				        },
				        formatter: function (params) {
			           		var res = params[0].axisValue + "<br/>" +params[0].marker + params[0].seriesName + ' : ';
			           		if(params[0].value == -1){
			           			res += '跌';
			           		}
			           		else if(params[0].value == 1){
			           			res += '涨';
			           		}
			           		else{
			           			res += '平';
			           		}
			           		return res;
			        }
				    },
				    grid: {
				    	top: '20%',
				        left: '3%',
				        right: '4%',
				        bottom: '8%',
				        containLabel: true
				    },
				    xAxis : [
				        {
				            type : 'category',
				            data : this.test_data.date,
				            axisTick: {
				                alignWithLabel: true
				            }
				        }
				    ],
				    yAxis : [
				        {
				            type : 'value',
				            interval : 1,
				            axisLabel: {
				                formatter: function (value, index) {
				                    if(value == 1) return '涨';
				                    else if(value == -1) return'跌';
				                    else if(value === 0) return '平';
				                }
				            }
				        }
				    ],
				    series : [
				        {
				            name:'预测走势',
				            type:'bar',
				            barWidth: '60%',
				            data:this.test_data.predtrend
				        }
				    ]
			});
		},

    	drawchartRealHistogram(){
			this.chartRealHistogram = echarts.init(document.getElementById('chartRealHistogram'));
			this.chartRealHistogram.setOption({
					color: ['#3398DB'],
					title: {
	                	left: 'center',
	                    text: '实际涨跌走势'
	                },
				    tooltip : {
				        trigger: 'axis',
				        axisPointer : {
				            type : 'shadow'
				        },
				        formatter: function (params) {
			           		var res = params[0].axisValue + "<br/>" +params[0].marker + params[0].seriesName + ' : ';
			           		if(params[0].value == -1){
			           			res += '跌';
			           		}
			           		else if(params[0].value == 1){
			           			res += '涨';
			           		}
			           		else{
			           			res += '平';
			           		}
			           		return res;
			        }
				    },
				    grid: {
				    	top: '20%',
				        left: '3%',
				        right: '4%',
				        bottom: '8%',
				        containLabel: true
				    },
				    xAxis : [
				        {
				            type : 'category',
				            data : this.test_data.date,
				            axisTick: {
				                alignWithLabel: true
				            }
				        }
				    ],
				    yAxis : [
				        {
				            type : 'value',
				            interval : 1,
				            axisLabel: {
				                formatter: function (value, index) {
				                    if(value == 1) return '涨';
				                    else if(value == -1) return'跌';
				                    else if(value === 0) return '平';
				                }
				            }
				        }
				    ],
				    series : [
				        {
				            name:'实际走势',
				            type:'bar',
				            barWidth: '60%',
				            data:this.test_data.realtrend
				        },
				    ]
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
                    bottom: '8%',
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
