个人的Python练习平台

# Git提交规范
在团队协作中使用Git进行版本控制时，`fix`、`feat`等是用于标识提交内容类型的约定标签，这些标签通常与Angular团队提出的Conventional Commits规范相关。这种规范旨在通过结构化的提交信息来简化变更管理、自动化生成更新日志等。

以下是几种常见的类型：

- `feat`: 表示一次提交为代码库添加了一个新特性，对应于语义化版本(SemVer)中的次要版本更新。
- `fix`: 表示修复了一个bug，这也对应于SemVer中的补丁版本更新。
- `docs`: 文档（documentation）的更新，比如修改README文件等。
- `style`: 不影响代码含义的修改，如格式化、去掉尾随空格、改变字符顺序等。
- `refactor`: 代码重构，既不修正错误也不添加新功能的代码更改。
- `perf`: 改进性能的代码更改。
- `test`: 增加或修正测试用例。
- `chore`: 更新grunt任务等；不影响源文件、测试文件的内容。
- `ci`: 修改CI配置文件和脚本（例如 Travis, Jenkins 等）。
- `build`: 影响构建系统或外部依赖项的修改（例如gulp, broccoli, npm等）。

按照Conventional Commits规范，一个标准的提交信息应该包含以下三个部分：类型(type)、可选的作用域(scope)以及描述(subject)。例如：

```
<type>(<scope>): <subject>
```

示例提交信息可能看起来像这样：

```
feat(api): add pagination to user list endpoint
```

这意味着这次提交为API添加了分页功能到用户列表端点，属于新特性的添加。

遵循这样的规范有助于提高团队成员之间的沟通效率，便于自动化工具识别并处理不同类型的提交。同时也有利于自动生成详细的变更日志，简化发布流程。不过需要注意的是，这只是一个社区推荐的实践，并不是Git强制要求的标准。因此，具体项目可能会根据自身需求对这些规则进行调整。

# Chrome浏览器驱动相关
当前版本：  
https://googlechromelabs.github.io/chrome-for-testing/
历史版本：  
https://chromedriver.storage.googleapis.com/index.html