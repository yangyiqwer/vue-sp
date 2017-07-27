import NotFound from './views/404.vue'
import Home from './views/Home.vue'
import train from './views/nav/train.vue'
import test from './views/nav/test.vue'
import pred from './views/nav/pred.vue'

let routes = [
    {
        path: '/404',
        component: NotFound,
        name: '',
        hidden: true
    },
    {
        path: '/',
        component: Home,
        name: '',
        iconCls: 'el-icon-edit',
        leaf: true,
        children: [
            { path: '/train', component: train, name: '训练' }
        ]
    },
    {
        path: '/',
        component: Home,
        name: '',
        iconCls: 'el-icon-search',
        leaf: true,
        children: [
            { path: '/test', component: test, name: '测试' }
        ]
    },
    {
        path: '/',
        component: Home,
        name: '',
        iconCls: 'fa fa-bar-chart',
        leaf: true,
        children: [
            { path: '/pred', component: pred, name: '预测' }
        ]
    },
    {
        path: '*',
        hidden: true,
        redirect: { path: '/404' }
    }
];

export default routes;