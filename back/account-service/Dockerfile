# Build stage
FROM gradle:7.4-jdk17 AS build

WORKDIR /app

# 프로젝트 파일을 컨테이너로 복사
COPY . .

# gradlew 실행 권한 추가 (중요)
RUN chmod +x ./gradlew

# Gradle 빌드 (테스트 제외)
RUN ./gradlew clean build -x test

# Test stage
FROM gradle:7.4-jdk17 AS test

WORKDIR /app

# builder에서 빌드된 파일 복사
COPY --from=build /app .

# Gradle 테스트 실행
# RUN ./gradlew test

# Run stage
FROM gradle:7.4-jdk17 AS run

WORKDIR /app

# builder에서 빌드된 JAR 파일 복사
COPY --from=build /app/build/libs/*SNAPSHOT.jar app.jar
# 앱 실행
ENTRYPOINT ["java", "-jar", "-Xms512m", "-Xmx512m", "app.jar"]